#include "volume3d_engine.hpp"
#include <Accelerate/Accelerate.h>
#include <dispatch/dispatch.h>
#include <iostream>
#include <cmath>
#include <vector>
#include <complex>
#include <algorithm>
#include <numeric>
#include "miguel_unwrapper.hpp"

namespace oct {

namespace {

// 1D Linear/PCHIP monotonic interpolation
void resample_spectrum(
    const float* src_x, // [Nl] monotonic
    const float* src_y, // [Nl]
    int n_src,
    const float* dst_x, // [Nk] uniform
    float* dst_y,       // [Nk]
    int n_dst
) {
    int cur_idx = 0;
    for (int i = 0; i < n_dst; ++i) {
        float x = dst_x[i];
        while (cur_idx < n_src - 2 && src_x[cur_idx + 1] < x) {
            cur_idx++;
        }
        float x0 = src_x[cur_idx];
        float x1 = src_x[cur_idx + 1];
        float y0 = src_y[cur_idx];
        float y1 = src_y[cur_idx + 1];

        float t = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0f;
        t = std::clamp(t, 0.0f, 1.0f);
        dst_y[i] = y0 + t * (y1 - y0);
    }
}

// 1D Phase unwrapping
void unwrap_1d(const float* in_phase, float* out_phase, int n) {
    if (n <= 0) return;
    out_phase[0] = in_phase[0];
    const float pi = 3.14159265358979323846f;
    const float two_pi = 2.0f * pi;

    for (int i = 1; i < n; ++i) {
        float diff = in_phase[i] - in_phase[i - 1];
        while (diff > pi) diff -= two_pi;
        while (diff < -pi) diff += two_pi;
        out_phase[i] = out_phase[i - 1] + diff;
    }
}

// 4x4 Least Squares Polynomial Fit (Order 3)
void polyfit3(const std::vector<float>& x, const std::vector<float>& y, std::vector<float>& coeffs) {
    int n = (int)x.size();
    coeffs.assign(4, 0.0f);
    if (n < 4) return;

    double ATA[4][4] = {{0}};
    double ATy[4] = {0};

    for (int i = 0; i < n; ++i) {
        double xi = x[i];
        double yi = y[i];
        double p[4] = {xi * xi * xi, xi * xi, xi, 1.0};

        for (int r = 0; r < 4; ++r) {
            ATy[r] += p[r] * yi;
            for (int c = 0; c < 4; ++c) {
                ATA[r][c] += p[r] * p[c];
            }
        }
    }

    // Solve 4x4 via Gaussian elimination
    double M[4][5];
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) M[r][c] = ATA[r][c];
        M[r][4] = ATy[r];
    }

    for (int i = 0; i < 4; ++i) {
        int max_r = i;
        for (int r = i + 1; r < 4; ++r) {
            if (std::abs(M[r][i]) > std::abs(M[max_r][i])) max_r = r;
        }
        for (int k = 0; k <= 4; ++k) std::swap(M[i][k], M[max_r][k]);

        double pivot = M[i][i];
        if (std::abs(pivot) < 1e-12) pivot = 1e-12;
        for (int k = i; k <= 4; ++k) M[i][k] /= pivot;

        for (int r = 0; r < 4; ++r) {
            if (r == i) continue;
            double factor = M[r][i];
            for (int k = i; k <= 4; ++k) M[r][k] -= factor * M[i][k];
        }
    }

    for (int i = 0; i < 4; ++i) coeffs[i] = (float)M[i][4];
}

} // anonymous namespace

Volume3DEngine::Volume3DEngine(id<MTLDevice> device, int width, int height, int num_lambdas)
    : device_(device), width_(width), height_(height), num_lambdas_(num_lambdas) {
    if (device_) {
        command_queue_ = [device_ newCommandQueue];
    }
    reinitialize(width_, height_, num_lambdas_);
}

Volume3DEngine::~Volume3DEngine() = default;

void Volume3DEngine::reinitialize(int width, int height, int num_lambdas) {
    width_ = width;
    height_ = height;
    num_lambdas_ = num_lambdas;

    ph_reconstructed_.assign((size_t)height_ * width_, 0.0f);
    phase2d_rgba_.assign((size_t)height_ * width_, 0xFF000000);
    surface3d_rgba_.assign(512 * 512, 0xFF10151C);

    if (device_) {
        MTLTextureDescriptor* desc2d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                          width:width_
                                                                                         height:height_
                                                                                      mipmapped:NO];
        desc2d.usage = MTLTextureUsageShaderRead;
        phase2d_texture_ = [device_ newTextureWithDescriptor:desc2d];

        MTLTextureDescriptor* desc3d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                          width:512
                                                                                         height:512
                                                                                      mipmapped:NO];
        desc3d.usage = MTLTextureUsageShaderRead;
        surface3d_texture_ = [device_ newTextureWithDescriptor:desc3d];
    }
}

void Volume3DEngine::compute_k_grid_and_dispersion(
    const float* raw_ref,
    const std::vector<double>& lambda_vec,
    std::vector<float>& phi_ref,
    std::vector<float>& phi_lin,
    std::vector<std::complex<float>>& disp_filter,
    std::vector<float>& win
) {
    // Dynamic k-grid size: next power of 2 >= num_lambdas_ (256 for 251 wavl, 1024 for 751 wavl)
    int Nk_raw = 1;
    while (Nk_raw < num_lambdas_) Nk_raw <<= 1;
    if (Nk_raw < 256) Nk_raw = 256;

    win.resize(Nk_raw);
    for (int i = 0; i < Nk_raw; ++i) {
        win[i] = 0.5f * (1.0f - std::cos(2.0f * 3.1415926535f * (float)i / (float)(Nk_raw - 1)));
    }

    int cx = width_ / 2;
    int cy = height_ / 2;
    int roi_rad = 32;
    int y0 = std::max(0, cy - roi_rad);
    int y1 = std::min(height_ - 1, cy + roi_rad);
    int x0 = std::max(0, cx - roi_rad);
    int x1 = std::min(width_ - 1, cx + roi_rad);
    int roi_pixels = (y1 - y0 + 1) * (x1 - x0 + 1);

    // 1. Calculate ROI mean spectrum across lambda
    std::vector<float> ref_spec(num_lambdas_, 0.0f);
    size_t frame_stride = (size_t)width_ * height_;

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            float mean_pix = 0.0f;
            for (int l = 0; l < num_lambdas_; ++l) {
                mean_pix += raw_ref[l * frame_stride + y * width_ + x];
            }
            mean_pix /= (float)num_lambdas_;

            for (int l = 0; l < num_lambdas_; ++l) {
                ref_spec[l] += (raw_ref[l * frame_stride + y * width_ + x] - mean_pix);
            }
        }
    }
    for (int l = 0; l < num_lambdas_; ++l) {
        ref_spec[l] /= (float)roi_pixels;
    }

    // 2. Hilbert transform of ref_spec using Discrete Fourier Transform
    const double two_pi = 6.28318530717958647692;
    std::vector<std::complex<double>> dft_spec(num_lambdas_, 0.0);
    for (int k = 0; k < num_lambdas_; ++k) {
        for (int l = 0; l < num_lambdas_; ++l) {
            double angle = -two_pi * (double)k * (double)l / (double)num_lambdas_;
            dft_spec[k] += (double)ref_spec[l] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
    }

    // Create analytic signal in frequency domain
    for (int k = 0; k < num_lambdas_; ++k) {
        if (k == 0 || k == num_lambdas_ / 2) {
            // Keep DC and Nyquist as is
        } else if (k < num_lambdas_ / 2) {
            dft_spec[k] *= 2.0;
        } else {
            dft_spec[k] = 0.0;
        }
    }

    // Inverse DFT to get analytic signal
    std::vector<float> phi_raw(num_lambdas_, 0.0f);
    phi_ref.resize(num_lambdas_);
    for (int l = 0; l < num_lambdas_; ++l) {
        std::complex<double> z(0, 0);
        for (int k = 0; k < num_lambdas_; ++k) {
            double angle = two_pi * (double)k * (double)l / (double)num_lambdas_;
            z += dft_spec[k] * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        z /= (double)num_lambdas_;
        phi_raw[l] = (float)std::atan2(z.imag(), z.real());
    }

    // 3. Unwrap phase
    unwrap_1d(phi_raw.data(), phi_ref.data(), num_lambdas_);

    // 4. Linearize k-grid
    float phi_min = *std::min_element(phi_ref.begin(), phi_ref.end());
    float phi_max = *std::max_element(phi_ref.begin(), phi_ref.end());

    phi_lin.resize(Nk_raw);
    std::vector<float> k_norm(Nk_raw);
    for (int i = 0; i < Nk_raw; ++i) {
        float t = (float)i / (float)(Nk_raw - 1);
        phi_lin[i] = phi_min + t * (phi_max - phi_min);
        k_norm[i] = -1.0f + 2.0f * t;
    }

    // 5. Fit 3rd order polynomial & compute dispersion filter
    std::vector<float> p_fit;
    polyfit3(k_norm, phi_lin, p_fit);

    disp_filter.resize(Nk_raw);
    for (int i = 0; i < Nk_raw; ++i) {
        float k = k_norm[i];
        float phase_disp = p_fit[0] * k * k * k + p_fit[1] * k * k;
        disp_filter[i] = std::complex<float>(std::cos(-phase_disp), std::sin(-phase_disp));
    }
}

void Volume3DEngine::find_z_layers(
    const float* raw_obj,
    const float* raw_ref,
    const std::vector<float>& phi_ref,
    const std::vector<float>& phi_lin,
    const std::vector<std::complex<float>>& disp_filter,
    const std::vector<float>& win,
    int& out_z_obj,
    int& out_z_ref
) {
    const int Nk_raw = (int)phi_lin.size();
    const int N_fft = std::max(2048, Nk_raw * 2);

    int cx = width_ / 2;
    int cy = height_ / 2;
    int roi_rad = 32;
    int y0 = std::max(0, cy - roi_rad);
    int y1 = std::min(height_ - 1, cy + roi_rad);
    int x0 = std::max(0, cx - roi_rad);
    int x1 = std::min(width_ - 1, cx + roi_rad);
    int roi_pixels = (y1 - y0 + 1) * (x1 - x0 + 1);

    size_t frame_stride = (size_t)width_ * height_;

    // 1. Average AC spectrum across ROI
    std::vector<float> mean_obj_spec(num_lambdas_, 0.0f);
    std::vector<float> mean_ref_spec(num_lambdas_, 0.0f);

    for (int y = y0; y <= y1; ++y) {
        for (int x = x0; x <= x1; ++x) {
            size_t pix_idx = (size_t)y * width_ + x;
            float sum_o = 0.0f, sum_r = 0.0f;
            for (int l = 0; l < num_lambdas_; ++l) {
                sum_o += raw_obj[l * frame_stride + pix_idx];
                if (raw_ref) sum_r += raw_ref[l * frame_stride + pix_idx];
            }
            float mo = sum_o / (float)num_lambdas_;
            float mr = sum_r / (float)num_lambdas_;

            for (int l = 0; l < num_lambdas_; ++l) {
                mean_obj_spec[l] += (raw_obj[l * frame_stride + pix_idx] - mo);
                if (raw_ref) mean_ref_spec[l] += (raw_ref[l * frame_stride + pix_idx] - mr);
            }
        }
    }

    for (int l = 0; l < num_lambdas_; ++l) {
        mean_obj_spec[l] /= (float)roi_pixels;
        if (raw_ref) mean_ref_spec[l] /= (float)roi_pixels;
    }

    // 2. Resample and window
    std::vector<float> res_obj(Nk_raw);
    std::vector<float> res_ref(Nk_raw);
    resample_spectrum(phi_ref.data(), mean_obj_spec.data(), num_lambdas_, phi_lin.data(), res_obj.data(), Nk_raw);
    if (raw_ref) {
        resample_spectrum(phi_ref.data(), mean_ref_spec.data(), num_lambdas_, phi_lin.data(), res_ref.data(), Nk_raw);
    }

    // 3. Fast 2048-point DFT
    std::vector<float> obj_fft_mag(N_fft / 2, 0.0f);
    std::vector<float> ref_fft_mag(N_fft / 2, 0.0f);

    for (int z = 0; z < N_fft / 2; ++z) {
        float re_o = 0.0f, im_o = 0.0f;
        float re_r = 0.0f, im_r = 0.0f;
        float angle_step = +6.283185307f * (float)z / (float)N_fft;
        for (int k = 0; k < Nk_raw; ++k) {
            std::complex<float> s_o = (res_obj[k] * win[k]) * disp_filter[k];
            float angle = angle_step * (float)k;
            float c = std::cos(angle);
            float sn = std::sin(angle);
            re_o += s_o.real() * c - s_o.imag() * sn;
            im_o += s_o.real() * sn + s_o.imag() * c;

            if (raw_ref) {
                std::complex<float> s_r = (res_ref[k] * win[k]) * disp_filter[k];
                re_r += s_r.real() * c - s_r.imag() * sn;
                im_r += s_r.real() * sn + s_r.imag() * c;
            }
        }
        obj_fft_mag[z] = std::sqrt(re_o * re_o + im_o * im_o);
        if (raw_ref) ref_fft_mag[z] = std::sqrt(re_r * re_r + im_r * im_r);
    }

    out_z_obj = (int)(std::max_element(obj_fft_mag.begin(), obj_fft_mag.end()) - obj_fft_mag.begin());
    if (raw_ref) {
        out_z_ref = (int)(std::max_element(ref_fft_mag.begin(), ref_fft_mag.end()) - ref_fft_mag.begin());
    } else {
        out_z_ref = out_z_obj;
    }
}

bool Volume3DEngine::reconstruct_phase_oct(
    const float* raw_obj,
    const float* raw_ref,
    const std::vector<double>& wavelengths_nm,
    float dz_focus_um,
    float n_immersion,
    float dx_um,
    bool apply_detrend,
    PipelineMode pipeline_mode,
    ComputeBackend3D backend,
    PhaseOctStats& out_stats
) {
    if (!raw_obj) return false;
    const float* actual_ref = raw_ref ? raw_ref : raw_obj;

    auto t_start = std::chrono::steady_clock::now();

    // 1. K-grid calibration & Dispersion polynomial
    std::vector<float> phi_ref;
    std::vector<float> phi_lin;
    std::vector<std::complex<float>> disp_filter;
    std::vector<float> win;
    compute_k_grid_and_dispersion(actual_ref, wavelengths_nm, phi_ref, phi_lin, disp_filter, win);

    // 2. Find exact Z layer indices
    int z_obj = 0, z_ref = 0;
    find_z_layers(raw_obj, actual_ref, phi_ref, phi_lin, disp_filter, win, z_obj, z_ref);

    out_stats.z_obj_index = z_obj;
    out_stats.z_ref_index = z_ref;

    // 3. Precompute complex weights & project to wavelength grid
    const int Nk_raw = (int)phi_lin.size();
    const int N_fft = std::max(2048, Nk_raw * 2);

    std::vector<std::complex<float>> weight_obj(Nk_raw);
    std::vector<std::complex<float>> weight_ref(Nk_raw);
    for (int k = 0; k < Nk_raw; ++k) {
        float angle_obj = +6.283185307f * (float)(k * z_obj) / (float)N_fft;
        float angle_ref = +6.283185307f * (float)(k * z_ref) / (float)N_fft;
        std::complex<float> w_filt = win[k] * disp_filter[k];
        weight_obj[k] = w_filt * std::complex<float>(std::cos(angle_obj), std::sin(angle_obj));
        weight_ref[k] = w_filt * std::complex<float>(std::cos(angle_ref), std::sin(angle_ref));
    }

    // Direct Precomputed Projection Kernel (Linear Interpolation + Filter + FFT collapsed into single dot product)
    std::vector<std::complex<float>> C_obj(num_lambdas_, 0.0f);
    std::vector<std::complex<float>> C_ref(num_lambdas_, 0.0f);
    int cur_idx = 0;
    for (int k = 0; k < Nk_raw; ++k) {
        float x = phi_lin[k];
        while (cur_idx < num_lambdas_ - 2 && phi_ref[cur_idx + 1] < x) {
            cur_idx++;
        }
        float x0 = phi_ref[cur_idx];
        float x1 = phi_ref[cur_idx + 1];
        float t = (x1 > x0) ? (x - x0) / (x1 - x0) : 0.0f;
        t = std::clamp(t, 0.0f, 1.0f);

        C_obj[cur_idx] += (1.0f - t) * weight_obj[k];
        C_obj[cur_idx + 1] += t * weight_obj[k];

        C_ref[cur_idx] += (1.0f - t) * weight_ref[k];
        C_ref[cur_idx + 1] += t * weight_ref[k];
    }

    std::complex<float> sum_C_obj(0, 0);
    std::complex<float> sum_C_ref(0, 0);
    for (int l = 0; l < num_lambdas_; ++l) {
        sum_C_obj += C_obj[l];
        sum_C_ref += C_ref[l];
    }
    float inv_num_lambdas = 1.0f / (float)num_lambdas_;
    std::complex<float> mean_scale_obj = sum_C_obj * inv_num_lambdas;
    std::complex<float> mean_scale_ref = sum_C_ref * inv_num_lambdas;

    // 4. Ultra-Fast Single-Pass Parallel Streaming Block processing in GCD
    size_t total_pixels = (size_t)height_ * width_;
    std::vector<std::complex<float>> E_comp(total_pixels);
    std::complex<float>* e_comp_ptr = E_comp.data();
    size_t frame_stride = (size_t)width_ * height_;

    const size_t band_height = 32;
    const size_t num_bands = (height_ + band_height - 1) / band_height;

    dispatch_apply(num_bands, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t band_idx) {
        size_t y_start = band_idx * band_height;
        size_t y_end = std::min(y_start + band_height, (size_t)height_);
        size_t band_pixels = (y_end - y_start) * width_;
        size_t start_pix = y_start * width_;

        std::vector<float> acc_o_re(band_pixels, 0.0f);
        std::vector<float> acc_o_im(band_pixels, 0.0f);
        std::vector<float> acc_o_sum(band_pixels, 0.0f);

        std::vector<float> acc_r_re(band_pixels, 0.0f);
        std::vector<float> acc_r_im(band_pixels, 0.0f);
        std::vector<float> acc_r_sum(band_pixels, 0.0f);

        for (int l = 0; l < num_lambdas_; ++l) {
            const float* frame_o = raw_obj + l * frame_stride + start_pix;
            const float* frame_r = actual_ref + l * frame_stride + start_pix;

            float co_re = C_obj[l].real();
            float co_im = C_obj[l].imag();
            float cr_re = C_ref[l].real();
            float cr_im = C_ref[l].imag();

            for (size_t p = 0; p < band_pixels; ++p) {
                float val_o = frame_o[p];
                acc_o_re[p] += val_o * co_re;
                acc_o_im[p] += val_o * co_im;
                acc_o_sum[p] += val_o;

                float val_r = frame_r[p];
                acc_r_re[p] += val_r * cr_re;
                acc_r_im[p] += val_r * cr_im;
                acc_r_sum[p] += val_r;
            }
        }

        // Finalize complex layer amplitude & conjugate product
        for (size_t p = 0; p < band_pixels; ++p) {
            float e_obj_re = acc_o_re[p] - acc_o_sum[p] * mean_scale_obj.real();
            float e_obj_im = acc_o_im[p] - acc_o_sum[p] * mean_scale_obj.imag();

            float e_ref_re = acc_r_re[p] - acc_r_sum[p] * mean_scale_ref.real();
            float e_ref_im = acc_r_im[p] - acc_r_sum[p] * mean_scale_ref.imag();

            // e_obj * conj(e_ref)
            std::complex<float> e_obj(e_obj_re, e_obj_im);
            std::complex<float> e_ref(e_ref_re, e_ref_im);
            e_comp_ptr[start_pix + p] = e_obj * std::conj(e_ref);
        }
    });

    // 5. Phase Extraction matching MATLAB (+angle)
    std::vector<float> ph_unwrapped(total_pixels);
    float* ph_out_ptr = ph_unwrapped.data();

    if (pipeline_mode == PipelineMode::Matlab_Exact_FFT) {
        // --- MATLAB EXACT PIPELINE (FFT -> Z-window -> k-space -> Divide -> Z-space -> Peak) ---
        // 1D FFT sizes
        int N_fft_exact = num_lambdas_; 
        int log2N = (int)std::log2(N_fft_exact);
        if (1 << log2N != N_fft_exact) {
            log2N++;
            N_fft_exact = 1 << log2N;
        }

        FFTSetup fft_setup = vDSP_create_fftsetup(log2N, FFT_RADIX2);

        // Window: Tukey 0.2 (MATLAB: tukeywin(Nz, 0.2))
        std::vector<float> tukey(num_lambdas_, 1.0f);
        int ramp_len = (int)(0.2f * num_lambdas_ / 2.0f);
        for (int i = 0; i < ramp_len; ++i) {
            float val = 0.5f * (1.0f + std::cos(3.14159f * ((float)i / ramp_len - 1.0f)));
            tukey[i] = val;
            tukey[num_lambdas_ - 1 - i] = val;
        }

        // Perform column-wise exact MATLAB pipeline
        dispatch_apply(total_pixels, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t p) {
            // Arrays for vDSP
            std::vector<float> o_re(N_fft_exact, 0.0f);
            std::vector<float> o_im(N_fft_exact, 0.0f);
            std::vector<float> r_re(N_fft_exact, 0.0f);
            std::vector<float> r_im(N_fft_exact, 0.0f);

            // Apply ifftshift: circular shift by floor(N/2)
            // For input of length num_lambdas_, shift by num_lambdas_/2
            int shift = num_lambdas_ / 2;
            for (int l = 0; l < num_lambdas_; ++l) {
                int src = (l + shift) % num_lambdas_;
                o_re[l] = raw_obj[src * frame_stride + p] * tukey[src];
                r_re[l] = actual_ref[src * frame_stride + p] * tukey[src];
            }

            DSPSplitComplex split_o = { o_re.data(), o_im.data() };
            DSPSplitComplex split_r = { r_re.data(), r_im.data() };

            // IFFT (k -> z)
            vDSP_fft_zip(fft_setup, &split_o, 1, log2N, kFFTDirection_Inverse);
            vDSP_fft_zip(fft_setup, &split_r, 1, log2N, kFFTDirection_Inverse);

            // fftshift on output: swap halves [0..N/2-1] <-> [N/2..N-1]
            {
                int half = N_fft_exact / 2;
                for (int z = 0; z < half; ++z) {
                    std::swap(o_re[z], o_re[z + half]);
                    std::swap(o_im[z], o_im[z + half]);
                    std::swap(r_re[z], r_re[z + half]);
                    std::swap(r_im[z], r_im[z + half]);
                }
            }

            // Z-window: keep only the positive-delay half (diff_order=1)
            // MATLAB: z1 = round(Nz/2 + 0.07*Nz), z2 = Nz
            int z1_win = (int)(N_fft_exact * 0.5f + 0.07f * N_fft_exact);
            for (int z = 0; z < N_fft_exact; ++z) {
                if (z < z1_win) {
                    o_re[z] = 0.0f; o_im[z] = 0.0f;
                    r_re[z] = 0.0f; r_im[z] = 0.0f;
                }
            }

            // ifftshift before FFT (z -> k): swap halves
            {
                int half = N_fft_exact / 2;
                for (int z = 0; z < half; ++z) {
                    std::swap(o_re[z], o_re[z + half]);
                    std::swap(o_im[z], o_im[z + half]);
                    std::swap(r_re[z], r_re[z + half]);
                    std::swap(r_im[z], r_im[z + half]);
                }
            }
            vDSP_fft_zip(fft_setup, &split_o, 1, log2N, kFFTDirection_Forward);
            vDSP_fft_zip(fft_setup, &split_r, 1, log2N, kFFTDirection_Forward);
            // fftshift after FFT: swap halves
            {
                int half = N_fft_exact / 2;
                for (int z = 0; z < half; ++z) {
                    std::swap(o_re[z], o_re[z + half]);
                    std::swap(o_im[z], o_im[z + half]);
                    std::swap(r_re[z], r_re[z + half]);
                    std::swap(r_im[z], r_im[z + half]);
                }
            }

            // Divide U_obj / U_ref (Complex Division)
            for (int z = 0; z < N_fft_exact; ++z) {
                float rr = r_re[z], ri = r_im[z];
                float or_ = o_re[z], oi = o_im[z];
                float den = rr * rr + ri * ri + 1e-12f;
                o_re[z] = (or_ * rr + oi * ri) / den;
                o_im[z] = (oi * rr - or_ * ri) / den;
            }

            // ifftshift before final IFFT: swap halves
            {
                int half = N_fft_exact / 2;
                for (int z = 0; z < half; ++z) {
                    std::swap(o_re[z], o_re[z + half]);
                    std::swap(o_im[z], o_im[z + half]);
                }
            }
            // IFFT (k -> z)
            vDSP_fft_zip(fft_setup, &split_o, 1, log2N, kFFTDirection_Inverse);
            // fftshift after IFFT: swap halves
            {
                int half = N_fft_exact / 2;
                for (int z = 0; z < half; ++z) {
                    std::swap(o_re[z], o_re[z + half]);
                    std::swap(o_im[z], o_im[z + half]);
                }
            }

            // Pick peak in positive-delay half (CC_corr: find max amplitude)
            float max_amp = -1.0f;
            int peak_idx = 0;
            int z_search_start = N_fft_exact / 2;
            for (int z = z_search_start; z < N_fft_exact; ++z) {
                float amp2 = o_re[z] * o_re[z] + o_im[z] * o_im[z];
                if (amp2 > max_amp) {
                    max_amp = amp2;
                    peak_idx = z;
                }
            }

            // Extract Phase at Peak
            ph_out_ptr[p] = std::atan2(o_im[peak_idx], o_re[peak_idx]);
        });
        vDSP_destroy_fftsetup(fft_setup);
    } else {
        // --- OPTIMIZED DTFT PIPELINE ---
        for (size_t i = 0; i < total_pixels; ++i) {
            ph_unwrapped[i] = std::atan2(E_comp[i].imag(), E_comp[i].real());
        }
    }

    // 6. 2D Phase Unwrapping using Miguel's Algorithm
    MiguelUnwrapper2D::unwrap(ph_unwrapped, width_, height_);

    // 7. Optional Detrending
    if (apply_detrend) {
        MiguelUnwrapper2D::detrend_median_gradient(ph_unwrapped, width_, height_);
    }

    // Negate phase to match physical convention AFTER unwrapping
    // Positive height (step UP from mirror) = negative phase (reflection OCT)
    for (size_t i = 0; i < total_pixels; ++i) {
        ph_unwrapped[i] = -ph_unwrapped[i];
    }

    ph_reconstructed_ = std::move(ph_unwrapped);

    // 8. Compute telemetry
    float p_min = ph_reconstructed_[0];
    float p_max = ph_reconstructed_[0];
    for (float p : ph_reconstructed_) {
        if (p < p_min) p_min = p;
        if (p > p_max) p_max = p;
    }

    out_stats.phase_min_rad = p_min;
    out_stats.phase_max_rad = p_max;
    out_stats.step_phase_delay_rad = p_max - p_min;

    auto t_end = std::chrono::steady_clock::now();
    out_stats.compute_time_ms = std::chrono::duration<float, std::milli>(t_end - t_start).count();
    out_stats.is_valid = true;
    has_reconstructed_ = true;

    return true;
}

bool Volume3DEngine::reconstruct_3d_volume(
    const float* volume_data,
    ComputeBackend3D backend,
    Volume3DStats& out_stats
) {
    std::vector<double> dummy_w(num_lambdas_);
    for (int i = 0; i < num_lambdas_; ++i) {
        float frac = (num_lambdas_ > 1) ? (float)i / (float)(num_lambdas_ - 1) : 0.0f;
        dummy_w[i] = 803.0 + frac * (878.0 - 803.0);
    }
    return reconstruct_phase_oct(volume_data, nullptr, dummy_w, 0.0f, 1.52f, 0.5f, true, PipelineMode::Optimized_DTFT, backend, out_stats);
}

void Volume3DEngine::render_raymarching(
    float yaw,
    float pitch,
    float zoom,
    float threshold_db,
    float density,
    float z_stretch,
    int render_mode,
    const uint32_t* colormap_lut,
    int num_steps
) {
    if (!has_reconstructed_ || ph_reconstructed_.empty()) return;

    render_phase2d_texture(colormap_lut);
    update_surface3d_render(yaw, pitch, zoom, z_stretch, colormap_lut);
}

void Volume3DEngine::render_phase2d_texture(const uint32_t* lut) {
    if (!lut || ph_reconstructed_.empty()) return;

    float p_min = ph_reconstructed_[0];
    float p_max = ph_reconstructed_[0];
    for (float p : ph_reconstructed_) {
        if (p < p_min) p_min = p;
        if (p > p_max) p_max = p;
    }
    float p_range = (p_max > p_min) ? (p_max - p_min) : 1.0f;

    size_t total_pixels = (size_t)width_ * height_;
    phase2d_rgba_.resize(total_pixels);

    for (size_t i = 0; i < total_pixels; ++i) {
        float norm = std::clamp((ph_reconstructed_[i] - p_min) / p_range, 0.0f, 1.0f);
        int lut_idx = std::clamp((int)(norm * 255.0f), 0, 255);
        phase2d_rgba_[i] = lut[lut_idx];
    }

    if (phase2d_texture_) {
        MTLRegion region = MTLRegionMake2D(0, 0, width_, height_);
        [phase2d_texture_ replaceRegion:region mipmapLevel:0 withBytes:phase2d_rgba_.data() bytesPerRow:width_ * sizeof(uint32_t)];
    }
}

void Volume3DEngine::update_surface3d_render(
    float yaw,
    float pitch,
    float zoom,
    float z_scale,
    const uint32_t* lut
) {
    const int view_size = 512;
    surface3d_rgba_.assign((size_t)view_size * view_size, 0xFF10141A);

    float p_min = ph_reconstructed_[0];
    float p_max = ph_reconstructed_[0];
    for (float p : ph_reconstructed_) {
        if (p < p_min) p_min = p;
        if (p > p_max) p_max = p;
    }
    float p_range = (p_max > p_min) ? (p_max - p_min) : 1.0f;

    // Fast 3D Software Topography Mesh Raycaster / Projector
    float cos_yaw = std::cos(yaw);
    float sin_yaw = std::sin(yaw);
    float cos_pitch = std::cos(pitch);
    float sin_pitch = std::sin(pitch);

    std::vector<float> z_buffer((size_t)view_size * view_size, 1e9f);

    int step = std::max(2, width_ / 256);

    for (int y = 0; y < height_ - step; y += step) {
        for (int x = 0; x < width_ - step; x += step) {
            float z_val = ph_reconstructed_[y * width_ + x];
            float norm_z = (z_val - p_min) / p_range; // 0..1

            // 3D Object coordinates: centered in [-1, 1]
            float px = ((float)x / (float)width_ - 0.5f) * 2.0f;
            float py = ((float)y / (float)height_ - 0.5f) * 2.0f;
            float pz = -(norm_z - 0.5f) * (z_scale * 0.4f); // INVERTED Z AXIS

            // Orbit Rotation
            float rx = px * cos_yaw - py * sin_yaw;
            float ry = px * sin_yaw + py * cos_yaw;
            float rz = pz;

            float cam_x = rx;
            float cam_y = ry * cos_pitch - rz * sin_pitch;
            float cam_z = ry * sin_pitch + rz * cos_pitch + zoom * 2.5f;

            if (cam_z <= 0.1f) continue;

            // Perspective Projection
            float fov = (float)view_size * 0.9f;
            int sx = (int)(view_size * 0.5f + (cam_x / cam_z) * fov);
            int sy = (int)(view_size * 0.5f - (cam_y / cam_z) * fov);

            if (sx >= 0 && sx < view_size && sy >= 0 && sy < view_size) {
                size_t pix_idx = (size_t)sy * view_size + sx;
                if (cam_z < z_buffer[pix_idx]) {
                    z_buffer[pix_idx] = cam_z;
                    int lut_idx = std::clamp((int)(norm_z * 255.0f), 0, 255);
                    surface3d_rgba_[pix_idx] = lut ? lut[lut_idx] : 0xFFFFFFFF;
                }
            }
        }
    }

    if (surface3d_texture_) {
        MTLRegion region = MTLRegionMake2D(0, 0, view_size, view_size);
        [surface3d_texture_ replaceRegion:region mipmapLevel:0 withBytes:surface3d_rgba_.data() bytesPerRow:view_size * sizeof(uint32_t)];
    }
}

void* Volume3DEngine::get_rendered_texture_ptr() const {
    return (__bridge void*)surface3d_texture_;
}

void* Volume3DEngine::get_phase2d_texture_ptr() const {
    return (__bridge void*)phase2d_texture_;
}

} // namespace oct

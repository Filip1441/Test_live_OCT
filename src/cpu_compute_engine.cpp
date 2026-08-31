#include "cpu_compute_engine.hpp"
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>

namespace oct {

class SgemmFftPlan {
public:
    explicit SgemmFftPlan(int num_lambdas)
        : num_lambdas_(num_lambdas), depth_bins_(num_lambdas / 2 + 1) {
        cos_table_.resize((size_t)depth_bins_ * num_lambdas_);
        sin_table_.resize((size_t)depth_bins_ * num_lambdas_);

        const double pi2 = 6.28318530717958647692;
        for (int k = 0; k < depth_bins_; ++k) {
            double k_norm = -pi2 * (double)k / (double)num_lambdas_;
            for (int l = 0; l < num_lambdas_; ++l) {
                double angle = k_norm * (double)l;
                cos_table_[(size_t)k * num_lambdas_ + l] = (float)std::cos(angle);
                sin_table_[(size_t)k * num_lambdas_ + l] = (float)std::sin(angle);
            }
        }
    }

    int num_lambdas() const { return num_lambdas_; }
    int depth_bins() const { return depth_bins_; }
    const float* cos_table() const { return cos_table_.data(); }
    const float* sin_table() const { return sin_table_.data(); }

private:
    int num_lambdas_;
    int depth_bins_;
    std::vector<float> cos_table_;
    std::vector<float> sin_table_;
};

CpuComputeEngine::CpuComputeEngine(int num_lambdas)
    : num_lambdas_(num_lambdas), depth_bins_(num_lambdas / 2 + 1) {
    window_.resize(num_lambdas_);
    for (int i = 0; i < num_lambdas_; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * 3.1415926535f * (float)i / (float)(num_lambdas_ - 1)));
    }
    plan_ = std::make_unique<SgemmFftPlan>(num_lambdas_);
}

CpuComputeEngine::~CpuComputeEngine() = default;

void CpuComputeEngine::compute_from_lines(
    const float* __restrict__ h_raw,
    const float* __restrict__ v_raw,
    const float* __restrict__ ascan_raw,
    int n_lambda,
    int width,
    int height,
    CrosshairReconResult& out_result
) {
    if (n_lambda != num_lambdas_ || !plan_) {
        num_lambdas_ = n_lambda;
        depth_bins_ = n_lambda / 2 + 1;
        window_.resize(num_lambdas_);
        for (int i = 0; i < num_lambdas_; ++i) {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * 3.1415926535f * (float)i / (float)(num_lambdas_ - 1)));
        }
        plan_ = std::make_unique<SgemmFftPlan>(num_lambdas_);
    }

    out_result.h_bscan.resize((size_t)depth_bins_ * width);
    out_result.v_bscan.resize((size_t)depth_bins_ * height);
    out_result.ascan.resize(depth_bins_);
    out_result.depth_axis.resize(depth_bins_);
    for (int k = 0; k < depth_bins_; ++k) {
        out_result.depth_axis[k] = (float)k;
    }

    // 1. Process Horizontal B-Scan: [N_lambda, Width] -> [Depth, Width]
    {
        std::vector<float> h_win((size_t)num_lambdas_ * width);
        // Detrend along lambda and apply window
        for (int x = 0; x < width; ++x) {
            float sum = 0.0f;
            for (int l = 0; l < num_lambdas_; ++l) {
                sum += h_raw[(size_t)l * width + x];
            }
            float mean = sum / (float)num_lambdas_;
            for (int l = 0; l < num_lambdas_; ++l) {
                h_win[(size_t)l * width + x] = (h_raw[(size_t)l * width + x] - mean) * window_[l];
            }
        }

        std::vector<float> h_re((size_t)depth_bins_ * width);
        std::vector<float> h_im((size_t)depth_bins_ * width);

        // cblas_sgemm: [Depth x N_lambda] * [N_lambda x Width] = [Depth x Width]
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    depth_bins_, width, num_lambdas_,
                    1.0f, plan_->cos_table(), num_lambdas_,
                    h_win.data(), width,
                    0.0f, h_re.data(), width);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    depth_bins_, width, num_lambdas_,
                    1.0f, plan_->sin_table(), num_lambdas_,
                    h_win.data(), width,
                    0.0f, h_im.data(), width);

        const float inv_ln10_20 = 8.685889638f;
        for (size_t i = 0; i < (size_t)depth_bins_ * width; ++i) {
            float mag = std::sqrt(h_re[i] * h_re[i] + h_im[i] * h_im[i]) + 1e-6f;
            out_result.h_bscan[i] = inv_ln10_20 * std::log(mag);
        }
    }

    // 2. Process Vertical B-Scan: [N_lambda, Height] -> [Height, Depth]
    {
        std::vector<float> v_win((size_t)num_lambdas_ * height);
        for (int y = 0; y < height; ++y) {
            float sum = 0.0f;
            for (int l = 0; l < num_lambdas_; ++l) {
                sum += v_raw[(size_t)l * height + y];
            }
            float mean = sum / (float)num_lambdas_;
            for (int l = 0; l < num_lambdas_; ++l) {
                v_win[(size_t)l * height + y] = (v_raw[(size_t)l * height + y] - mean) * window_[l];
            }
        }

        std::vector<float> v_re((size_t)depth_bins_ * height);
        std::vector<float> v_im((size_t)depth_bins_ * height);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    depth_bins_, height, num_lambdas_,
                    1.0f, plan_->cos_table(), num_lambdas_,
                    v_win.data(), height,
                    0.0f, v_re.data(), height);

        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    depth_bins_, height, num_lambdas_,
                    1.0f, plan_->sin_table(), num_lambdas_,
                    v_win.data(), height,
                    0.0f, v_im.data(), height);

        const float inv_ln10_20 = 8.685889638f;
        for (int y = 0; y < height; ++y) {
            for (int k = 0; k < depth_bins_; ++k) {
                size_t src_idx = (size_t)k * height + y;
                float mag = std::sqrt(v_re[src_idx] * v_re[src_idx] + v_im[src_idx] * v_im[src_idx]) + 1e-6f;
                // Transposed layout for vertical B-scan: [Height, Depth]
                out_result.v_bscan[(size_t)y * depth_bins_ + k] = inv_ln10_20 * std::log(mag);
            }
        }
    }

    // 3. Process 1D A-Scan: [N_lambda] -> [Depth]
    {
        float sum = 0.0f;
        for (int l = 0; l < num_lambdas_; ++l) sum += ascan_raw[l];
        float mean = sum / (float)num_lambdas_;

        std::vector<float> a_win(num_lambdas_);
        for (int l = 0; l < num_lambdas_; ++l) {
            a_win[l] = (ascan_raw[l] - mean) * window_[l];
        }

        std::vector<float> a_re(depth_bins_);
        std::vector<float> a_im(depth_bins_);

        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    depth_bins_, num_lambdas_,
                    1.0f, plan_->cos_table(), num_lambdas_,
                    a_win.data(), 1,
                    0.0f, a_re.data(), 1);

        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    depth_bins_, num_lambdas_,
                    1.0f, plan_->sin_table(), num_lambdas_,
                    a_win.data(), 1,
                    0.0f, a_im.data(), 1);

        const float inv_ln10_20 = 8.685889638f;
        for (int k = 0; k < depth_bins_; ++k) {
            float mag = std::sqrt(a_re[k] * a_re[k] + a_im[k] * a_im[k]) + 1e-6f;
            out_result.ascan[k] = inv_ln10_20 * std::log(mag);
        }
    }
}

} // namespace oct

#include <metal_stdlib>
using namespace metal;

// Metal Compute Kernel: Reconstruct 2D C-Scan slice at specific depth Z (or full volume column)
kernel void oct_cscan_kernel(
    device const float* sweep_volume [[buffer(0)]], // [N_lambda, H, W] (751 x 1504 x 1504)
    device const float* window       [[buffer(1)]], // [N_lambda]
    device float*       out_cscan    [[buffer(2)]], // [H, W] (1504 x 1504)
    constant uint&      n_lambda     [[buffer(3)]], // 751
    constant uint&      depth_z      [[buffer(4)]], // target depth k (0..375)
    constant uint&      width        [[buffer(5)]], // 1504
    constant uint&      height       [[buffer(6)]], // 1504
    uint2               pos          [[thread_position_in_grid]]
) {
    uint x = pos.x;
    uint y = pos.y;

    if (x >= width || y >= height) return;

    size_t frame_stride = (size_t)width * height;
    size_t pixel_offset = (size_t)y * width + x;

    // 1. Calculate DC mean for this pixel
    float sum = 0.0f;
    for (uint l = 0; l < n_lambda; ++l) {
        sum += sweep_volume[l * frame_stride + pixel_offset];
    }
    float mean = sum / float(n_lambda);

    // 2. Real-DFT for target depth k = depth_z
    float re = 0.0f;
    float im = 0.0f;
    float k_factor = -6.28318530717958647692f * float(depth_z) / float(n_lambda);

    for (uint l = 0; l < n_lambda; ++l) {
        float val = (sweep_volume[l * frame_stride + pixel_offset] - mean) * window[l];
        float angle = k_factor * float(l);
        re += val * cos(angle);
        im += val * sin(angle);
    }

    float mag = sqrt(re * re + im * im) + 1e-6f;
    float db = 8.685889638f * log(mag);

    out_cscan[pixel_offset] = db;
}

// Metal Compute Kernel: 3D Maximum Intensity Projection (MIP) across depth range [z_start, z_end]
kernel void oct_mip_kernel(
    device const float* sweep_volume [[buffer(0)]], // [N_lambda, H, W]
    device const float* window       [[buffer(1)]], // [N_lambda]
    device float*       out_mip      [[buffer(2)]], // [H, W]
    constant uint&      n_lambda     [[buffer(3)]], // 751
    constant uint&      z_start      [[buffer(4)]], // e.g. 0
    constant uint&      z_end        [[buffer(5)]], // e.g. 376
    constant uint&      width        [[buffer(6)]], // 1504
    constant uint&      height       [[buffer(7)]], // 1504
    uint2               pos          [[thread_position_in_grid]]
) {
    uint x = pos.x;
    uint y = pos.y;

    if (x >= width || y >= height) return;

    size_t frame_stride = (size_t)width * height;
    size_t pixel_offset = (size_t)y * width + x;

    float sum = 0.0f;
    for (uint l = 0; l < n_lambda; ++l) {
        sum += sweep_volume[l * frame_stride + pixel_offset];
    }
    float mean = sum / float(n_lambda);

    float max_db = 0.0f;

    for (uint k = z_start; k < z_end; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        float k_factor = -6.28318530717958647692f * float(k) / float(n_lambda);

        for (uint l = 0; l < n_lambda; ++l) {
            float val = (sweep_volume[l * frame_stride + pixel_offset] - mean) * window[l];
            float angle = k_factor * float(l);
            re += val * cos(angle);
            im += val * sin(angle);
        }

        float mag = sqrt(re * re + im * im) + 1e-6f;
        float db = 8.685889638f * log(mag);
        if (db > max_db) {
            max_db = db;
        }
    }

    out_mip[pixel_offset] = max_db;
}

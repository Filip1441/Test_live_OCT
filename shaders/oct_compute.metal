#include <metal_stdlib>
using namespace metal;

// Real-DFT / FFT Metal Compute Kernel on Apple Silicon GPU
kernel void oct_fft_db_kernel(
    device const float* raw_input   [[buffer(0)]], // [N_lambda, M_lines]
    device const float* window      [[buffer(1)]], // [N_lambda]
    device float*       out_db      [[buffer(2)]], // H-Bscan: [Depth, Width], V-Bscan: [Height, Depth]
    constant uint&      n_lambda    [[buffer(3)]], // 751
    constant uint&      depth_bins  [[buffer(4)]], // 376
    constant uint&      m_lines     [[buffer(5)]], // 1504
    constant uint&      is_vertical [[buffer(6)]], // 1 for V-Bscan [Height, Depth], 0 for H-Bscan [Depth, Width]
    uint2               thread_pos  [[thread_position_in_grid]]
) {
    uint line_idx = thread_pos.x; // width (0..1503) or height (0..1503)
    uint k = thread_pos.y;        // depth (0..375)

    if (line_idx >= m_lines || k >= depth_bins) return;

    // 1. Calculate line mean for DC offset removal
    float sum = 0.0f;
    for (uint l = 0; l < n_lambda; ++l) {
        sum += raw_input[l * m_lines + line_idx];
    }
    float mean = sum / float(n_lambda);

    // 2. Real-DFT along lambda dimension
    float re = 0.0f;
    float im = 0.0f;
    float k_factor = -6.28318530717958647692f * float(k) / float(n_lambda);

    for (uint l = 0; l < n_lambda; ++l) {
        float val = (raw_input[l * m_lines + line_idx] - mean) * window[l];
        float angle = k_factor * float(l);
        re += val * cos(angle);
        im += val * sin(angle);
    }

    // 3. Logarithmic Magnitude in dB
    float mag = sqrt(re * re + im * im) + 1e-6f;
    float db = 8.685889638f * log(mag); // 20 * log10(mag) = 20 / ln(10) * ln(mag)

    if (is_vertical == 1) {
        // [Height, Depth_bins] -> row line_idx (Y in 0..1503), col k (depth in 0..375)
        out_db[line_idx * depth_bins + k] = db;
    } else {
        // [Depth_bins, Width] -> row k (depth in 0..375), col line_idx (X in 0..1503)
        out_db[k * m_lines + line_idx] = db;
    }
}

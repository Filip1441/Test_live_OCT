#pragma once

#include "types.hpp"
#include "config.hpp"

namespace oct {

struct CrosshairReconResult {
    std::vector<float> h_bscan;        // [Width * Depth] (1504 x 376)
    std::vector<float> v_bscan;        // [Depth * Height] (376 x 1504, aligned along Y)
    std::vector<float> ascan;          // [Depth] (376)
    std::vector<float> depth_axis;     // [Depth] (376)
};

class IComputeEngine {
public:
    virtual ~IComputeEngine() = default;

    virtual ComputeBackend get_backend_type() const = 0;
    virtual bool is_available() const { return true; }

    /**
     * Executes Fourier Transform and dB conversion on contiguous line buffers.
     * h_raw: [N_lambda * Width] (751 x 1504)
     * v_raw: [N_lambda * Height] (751 x 1504)
     * ascan_raw: [N_lambda] (751)
     */
    virtual void compute_from_lines(
        const float* __restrict__ h_raw,
        const float* __restrict__ v_raw,
        const float* __restrict__ ascan_raw,
        int n_lambda,
        int width,
        int height,
        CrosshairReconResult& out_result
    ) = 0;
};

std::unique_ptr<IComputeEngine> create_metal_compute_engine(int num_lambdas = SWEEP_POINTS);

} // namespace oct

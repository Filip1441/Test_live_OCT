#pragma once

#include "compute_engine.hpp"
#include <memory>
#include <vector>

namespace oct {

class CudaComputeEngine : public IComputeEngine {
public:
    explicit CudaComputeEngine(int num_lambdas = 751, int width = 1504, int height = 1504);
    ~CudaComputeEngine() override;

    void compute_from_lines(
        const float* h_line_buffer,
        const float* v_line_buffer,
        const float* ascan_line_buffer,
        int num_lambdas,
        int width,
        int height,
        CrosshairReconResult& out_result
    ) override;

    bool is_available() const override { return is_initialized_; }
    const char* get_name() const override { return "NVIDIA CUDA (RTX 4090)"; }

private:
    int num_lambdas_;
    int depth_bins_;
    int width_;
    int height_;
    bool is_initialized_ = false;
    void* impl_ = nullptr; // Opaque CUDA context pointer
};

} // namespace oct

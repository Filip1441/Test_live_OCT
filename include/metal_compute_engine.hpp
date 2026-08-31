#pragma once

#include "compute_engine.hpp"
#import <Metal/Metal.h>
#include <vector>

namespace oct {

class MetalComputeEngine : public IComputeEngine {
public:
    explicit MetalComputeEngine(int num_lambdas = SWEEP_POINTS);
    ~MetalComputeEngine() override;

    ComputeBackend get_backend_type() const override { return ComputeBackend::MetalGPU; }
    bool is_available() const override { return device_ != nil && pipeline_ != nil; }

    void compute_from_lines(
        const float* __restrict__ h_raw,
        const float* __restrict__ v_raw,
        const float* __restrict__ ascan_raw,
        int n_lambda,
        int width,
        int height,
        CrosshairReconResult& out_result
    ) override;

private:
    void reallocate_buffers(int n_lambda, int width, int height);

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> command_queue_ = nil;
    id<MTLComputePipelineState> pipeline_ = nil;

    int num_lambdas_ = 0;
    int width_ = 0;
    int height_ = 0;
    int depth_bins_ = 0;

    std::vector<float> window_;
    id<MTLBuffer> window_buffer_ = nil;

    id<MTLBuffer> h_in_buffer_ = nil;
    id<MTLBuffer> h_out_buffer_ = nil;

    id<MTLBuffer> v_in_buffer_ = nil;
    id<MTLBuffer> v_out_buffer_ = nil;
};

} // namespace oct

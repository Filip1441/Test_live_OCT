#include "metal_compute_engine.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>

namespace oct {

MetalComputeEngine::MetalComputeEngine(int num_lambdas)
    : num_lambdas_(num_lambdas), depth_bins_(num_lambdas / 2 + 1) {
    device_ = MTLCreateSystemDefaultDevice();
    if (!device_) {
        std::cerr << "[MetalComputeEngine] Error: Metal is not supported on this device." << std::endl;
        return;
    }

    command_queue_ = [device_ newCommandQueue];

    // Load and compile shader
    std::ifstream file("shaders/oct_compute.metal");
    std::string source;
    if (file.is_open()) {
        std::stringstream ss;
        ss << file.rdbuf();
        source = ss.str();
    } else {
        std::cerr << "[MetalComputeEngine] Error: Could not read shaders/oct_compute.metal" << std::endl;
        return;
    }

    NSError* error = nil;
    NSString* srcStr = [NSString stringWithUTF8String:source.c_str()];
    MTLCompileOptions* options = [[MTLCompileOptions alloc] init];
    options.fastMathEnabled = YES;
    id<MTLLibrary> library = [device_ newLibraryWithSource:srcStr options:options error:&error];

    if (!library) {
        std::cerr << "[MetalComputeEngine] Compilation error: " 
                  << [error.localizedDescription UTF8String] << std::endl;
        return;
    }

    id<MTLFunction> kernelFunc = [library newFunctionWithName:@"oct_fft_db_kernel"];
    pipeline_ = [device_ newComputePipelineStateWithFunction:kernelFunc error:&error];

    if (!pipeline_) {
        std::cerr << "[MetalComputeEngine] Pipeline state error: " 
                  << [error.localizedDescription UTF8String] << std::endl;
        return;
    }

    reallocate_buffers(num_lambdas_, RAW_WIDTH, RAW_HEIGHT);
    std::cout << "[MetalComputeEngine] Metal GPU Compute Engine initialized on: " 
              << [[device_ name] UTF8String] << std::endl;
}

MetalComputeEngine::~MetalComputeEngine() = default;

void MetalComputeEngine::reallocate_buffers(int n_lambda, int width, int height) {
    num_lambdas_ = n_lambda;
    depth_bins_ = n_lambda / 2 + 1;
    width_ = width;
    height_ = height;

    window_.resize(num_lambdas_);
    for (int i = 0; i < num_lambdas_; ++i) {
        window_[i] = 0.5f * (1.0f - std::cos(2.0f * 3.1415926535f * (float)i / (float)(num_lambdas_ - 1)));
    }

    window_buffer_ = [device_ newBufferWithBytes:window_.data()
                                          length:num_lambdas_ * sizeof(float)
                                         options:MTLResourceStorageModeShared];

    size_t h_in_size = (size_t)num_lambdas_ * width_ * sizeof(float);
    size_t h_out_size = (size_t)depth_bins_ * width_ * sizeof(float);
    size_t v_in_size = (size_t)num_lambdas_ * height_ * sizeof(float);
    size_t v_out_size = (size_t)depth_bins_ * height_ * sizeof(float);

    h_in_buffer_ = [device_ newBufferWithLength:h_in_size options:MTLResourceStorageModeShared];
    h_out_buffer_ = [device_ newBufferWithLength:h_out_size options:MTLResourceStorageModeShared];
    v_in_buffer_ = [device_ newBufferWithLength:v_in_size options:MTLResourceStorageModeShared];
    v_out_buffer_ = [device_ newBufferWithLength:v_out_size options:MTLResourceStorageModeShared];
}

void MetalComputeEngine::compute_from_lines(
    const float* __restrict__ h_raw,
    const float* __restrict__ v_raw,
    const float* __restrict__ ascan_raw,
    int n_lambda,
    int width,
    int height,
    CrosshairReconResult& out_result
) {
    if (!is_available()) return;

    if (n_lambda != num_lambdas_ || width != width_ || height != height_) {
        reallocate_buffers(n_lambda, width, height);
    }

    out_result.h_bscan.resize((size_t)depth_bins_ * width_);
    out_result.v_bscan.resize((size_t)depth_bins_ * height_);
    out_result.ascan.resize(depth_bins_);
    out_result.depth_axis.resize(depth_bins_);
    for (int k = 0; k < depth_bins_; ++k) {
        out_result.depth_axis[k] = (float)k;
    }

    // Copy input line buffers to shared Metal buffers
    std::memcpy([h_in_buffer_ contents], h_raw, (size_t)num_lambdas_ * width_ * sizeof(float));
    std::memcpy([v_in_buffer_ contents], v_raw, (size_t)num_lambdas_ * height_ * sizeof(float));

    id<MTLCommandBuffer> commandBuffer = [command_queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];

    [encoder setComputePipelineState:pipeline_];
    [encoder setBuffer:window_buffer_ offset:0 atIndex:1];

    uint32_t nl = (uint32_t)num_lambdas_;
    uint32_t db = (uint32_t)depth_bins_;
    uint32_t w = (uint32_t)width_;
    uint32_t h = (uint32_t)height_;
    uint32_t is_vert = 0;

    [encoder setBytes:&nl length:sizeof(uint32_t) atIndex:3];
    [encoder setBytes:&db length:sizeof(uint32_t) atIndex:4];

    // 1. Dispatch H-Bscan Pass: grid (Width, Depth)
    [encoder setBuffer:h_in_buffer_ offset:0 atIndex:0];
    [encoder setBuffer:h_out_buffer_ offset:0 atIndex:2];
    [encoder setBytes:&w length:sizeof(uint32_t) atIndex:5];
    is_vert = 0;
    [encoder setBytes:&is_vert length:sizeof(uint32_t) atIndex:6];

    MTLSize threadgroupSize = MTLSizeMake(16, 16, 1);
    MTLSize gridSizeH = MTLSizeMake(width_, depth_bins_, 1);
    [encoder dispatchThreads:gridSizeH threadsPerThreadgroup:threadgroupSize];

    // 2. Dispatch V-Bscan Pass: grid (Height, Depth)
    [encoder setBuffer:v_in_buffer_ offset:0 atIndex:0];
    [encoder setBuffer:v_out_buffer_ offset:0 atIndex:2];
    [encoder setBytes:&h length:sizeof(uint32_t) atIndex:5];
    is_vert = 1;
    [encoder setBytes:&is_vert length:sizeof(uint32_t) atIndex:6];

    MTLSize gridSizeV = MTLSizeMake(height_, depth_bins_, 1);
    [encoder dispatchThreads:gridSizeV threadsPerThreadgroup:threadgroupSize];

    [encoder endEncoding];
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];

    // Copy results back
    std::memcpy(out_result.h_bscan.data(), [h_out_buffer_ contents], (size_t)depth_bins_ * width_ * sizeof(float));
    std::memcpy(out_result.v_bscan.data(), [v_out_buffer_ contents], (size_t)depth_bins_ * height_ * sizeof(float));

    // Fast 1D A-Scan on CPU
    float a_sum = 0.0f;
    for (int l = 0; l < num_lambdas_; ++l) a_sum += ascan_raw[l];
    float a_mean = a_sum / (float)num_lambdas_;

    const float inv_ln10_20 = 8.685889638f;
    for (int k = 0; k < depth_bins_; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        float k_factor = -6.283185307f * (float)k / (float)num_lambdas_;
        for (int l = 0; l < num_lambdas_; ++l) {
            float val = (ascan_raw[l] - a_mean) * window_[l];
            float angle = k_factor * (float)l;
            re += val * std::cos(angle);
            im += val * std::sin(angle);
        }
        float mag = std::sqrt(re * re + im * im) + 1e-6f;
        out_result.ascan[k] = inv_ln10_20 * std::log(mag);
    }
}

std::unique_ptr<IComputeEngine> create_metal_compute_engine(int num_lambdas) {
    return std::make_unique<MetalComputeEngine>(num_lambdas);
}

} // namespace oct

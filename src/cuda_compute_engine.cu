#ifdef __CUDACC__
#include "cuda_compute_engine.hpp"
#include <cuda_runtime.h>
#include <cufft.h>
#include <iostream>
#include <cmath>

namespace oct {

__global__ void oct_cuda_ascan_kernel(
    const float* __restrict__ in_lines,
    const float* __restrict__ window,
    float* __restrict__ out_bscan,
    int num_lambdas,
    int num_pixels,
    int depth_bins
) {
    int pixel_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel_idx >= num_pixels) return;

    // 1. Compute DC Mean
    float sum = 0.0f;
    for (int l = 0; l < num_lambdas; ++l) {
        sum += in_lines[l * num_pixels + pixel_idx];
    }
    float mean = sum / float(num_lambdas);

    // 2. Real-DFT for all depth bins k
    for (int k = 0; k < depth_bins; ++k) {
        float re = 0.0f;
        float im = 0.0f;
        float k_factor = -6.28318530717958647692f * float(k) / float(num_lambdas);

        for (int l = 0; l < num_lambdas; ++l) {
            float val = (in_lines[l * num_pixels + pixel_idx] - mean) * window[l];
            float angle = k_factor * float(l);
            re += val * cosf(angle);
            im += val * sinf(angle);
        }

        float mag = sqrtf(re * re + im * im) + 1e-6f;
        float db = 8.685889638f * logf(mag);

        out_bscan[k * num_pixels + pixel_idx] = db;
    }
}

struct CudaImpl {
    float* d_window = nullptr;
    float* d_h_lines = nullptr;
    float* d_v_lines = nullptr;
    float* d_a_lines = nullptr;

    float* d_h_bscan = nullptr;
    float* d_v_bscan = nullptr;
    float* d_ascan = nullptr;
};

CudaComputeEngine::CudaComputeEngine(int num_lambdas, int width, int height)
    : num_lambdas_(num_lambdas), depth_bins_(num_lambdas / 2 + 1), width_(width), height_(height) {

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) return;

    cudaSetDevice(0);
    auto* impl = new CudaImpl();

    std::vector<float> h_window(num_lambdas_);
    for (int i = 0; i < num_lambdas_; ++i) {
        h_window[i] = 0.5f * (1.0f - cosf(2.0f * 3.1415926535f * float(i) / float(num_lambdas_ - 1)));
    }

    cudaMalloc(&impl->d_window, num_lambdas_ * sizeof(float));
    cudaMemcpy(impl->d_window, h_window.data(), num_lambdas_ * sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc(&impl->d_h_lines, (size_t)num_lambdas_ * width_ * sizeof(float));
    cudaMalloc(&impl->d_v_lines, (size_t)num_lambdas_ * height_ * sizeof(float));
    cudaMalloc(&impl->d_a_lines, (size_t)num_lambdas_ * sizeof(float));

    cudaMalloc(&impl->d_h_bscan, (size_t)depth_bins_ * width_ * sizeof(float));
    cudaMalloc(&impl->d_v_bscan, (size_t)depth_bins_ * height_ * sizeof(float));
    cudaMalloc(&impl->d_ascan, (size_t)depth_bins_ * sizeof(float));

    impl_ = impl;
    is_initialized_ = true;
    std::cout << "[CudaComputeEngine] NVIDIA CUDA Engine initialized on GPU." << std::endl;
}

CudaComputeEngine::~CudaComputeEngine() {
    if (impl_) {
        auto* impl = (CudaImpl*)impl_;
        cudaFree(impl->d_window);
        cudaFree(impl->d_h_lines);
        cudaFree(impl->d_v_lines);
        cudaFree(impl->d_a_lines);
        cudaFree(impl->d_h_bscan);
        cudaFree(impl->d_v_bscan);
        cudaFree(impl->d_ascan);
        delete impl;
        impl_ = nullptr;
    }
}

void CudaComputeEngine::compute_from_lines(
    const float* h_line_buffer,
    const float* v_line_buffer,
    const float* ascan_line_buffer,
    int num_lambdas,
    int width,
    int height,
    CrosshairReconResult& out_result
) {
    if (!impl_ || !is_initialized_) return;
    auto* impl = (CudaImpl*)impl_;

    out_result.h_bscan.resize((size_t)depth_bins_ * width);
    out_result.v_bscan.resize((size_t)depth_bins_ * height);
    out_result.ascan.resize(depth_bins_);

    // 1. Upload line buffers to GPU
    cudaMemcpy(impl->d_h_lines, h_line_buffer, (size_t)num_lambdas * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(impl->d_v_lines, v_line_buffer, (size_t)num_lambdas * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(impl->d_a_lines, ascan_line_buffer, (size_t)num_lambdas * sizeof(float), cudaMemcpyHostToDevice);

    // 2. Launch CUDA kernels across RTX 4090 multiprocessors
    int block_size = 256;

    int grid_h = (width + block_size - 1) / block_size;
    oct_cuda_ascan_kernel<<<grid_h, block_size>>>(
        impl->d_h_lines, impl->d_window, impl->d_h_bscan, num_lambdas, width, depth_bins_
    );

    int grid_v = (height + block_size - 1) / block_size;
    oct_cuda_ascan_kernel<<<grid_v, block_size>>>(
        impl->d_v_lines, impl->d_window, impl->d_v_bscan, num_lambdas, height, depth_bins_
    );

    oct_cuda_ascan_kernel<<<1, 1>>>(
        impl->d_a_lines, impl->d_window, impl->d_ascan, num_lambdas, 1, depth_bins_
    );

    // 3. Download results
    cudaMemcpy(out_result.h_bscan.data(), impl->d_h_bscan, (size_t)depth_bins_ * width * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_result.v_bscan.data(), impl->d_v_bscan, (size_t)depth_bins_ * height * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(out_result.ascan.data(), impl->d_ascan, (size_t)depth_bins_ * sizeof(float), cudaMemcpyDeviceToHost);
}

} // namespace oct
#endif

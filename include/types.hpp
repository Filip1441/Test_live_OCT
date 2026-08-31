#pragma once

#include <vector>
#include <cstdint>
#include <string>

namespace oct {

enum class ProcessingApproach {
    RollingBuffer = 0,
    FullSweep = 1
};

enum class ComputeBackend {
    CPU = 0,
    MetalGPU = 1,
    CUDA = 2
};

struct DatasetMetadata {
    float refractive_index = 1.33f;
    float na = 1.3f;
    std::string raw_dtype = "float32";
    std::string sample_name = "SS-FF-OCT Sample";
};

struct Dataset {
    std::string uuid;
    int num_wavelengths = 751;
    int height = 1504;
    int width = 1504;
    float lambda_step_nm = 0.1f;
    std::vector<double> wavelengths_nm;
    std::vector<float> data; // [N_lambda, Height, Width]
    DatasetMetadata metadata;

    const float* get_frame(int lambda_idx) const {
        if (lambda_idx < 0 || lambda_idx >= num_wavelengths) return nullptr;
        return data.data() + (size_t)lambda_idx * height * width;
    }
};

struct CrosshairPos {
    int x = 752;
    int y = 752;
};

struct RawFrameMessage {
    const float* frame_ptr = nullptr; // Direct pointer into resident volume or scratch
    std::vector<float> frame_data;     // Owning buffer if synthetic or corrupted
    int lambda_index = 0;
    float wavelength_nm = 803.0f;
    int sweep_index = 0;
    uint64_t timestamp_ns = 0;
    bool is_dropped = false;
    bool is_corrupt = false;
    bool is_duplicate = false;
};

struct ProcessedResult {
    std::vector<float> camera_frame;   // [Height * Width]
    std::vector<float> h_bscan;        // [Width * Depth]
    std::vector<float> v_bscan;        // [Depth * Height] (aligned along Y)
    std::vector<float> ascan;          // [Depth]
    std::vector<float> depth_axis;     // [Depth]
    CrosshairPos crosshair;
    int sweep_index = 0;
    int lambda_index = 0;
    float producer_fps = 0.0f;
    float processor_fps = 0.0f;
    float compute_latency_ms = 0.0f;
    float buffer_fill_ratio = 0.0f;
    ProcessingApproach approach = ProcessingApproach::RollingBuffer;
    ComputeBackend backend = ComputeBackend::CPU;
};

struct ProducerStats {
    float fps = 0.0f;
    uint64_t frames_emitted = 0;
    uint64_t frames_dropped = 0;
    int current_sweep = 0;
    int current_lambda_idx = 0;
    float current_wavelength_nm = 803.0f;
};

struct ProcessorStats {
    float fps = 0.0f;
    uint64_t frames_processed = 0;
    float last_compute_latency_ms = 0.0f;
    float buffer_fill_ratio = 0.0f;
};

} // namespace oct

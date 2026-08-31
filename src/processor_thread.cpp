#include "processor_thread.hpp"
#include "cpu_compute_engine.hpp"
#include <chrono>
#include <iostream>
#include <cstring>
#include <cmath>

namespace oct {

ProcessorThread::ProcessorThread(
    std::shared_ptr<ThreadSafeQueue<RawFrameMessage>> raw_frame_queue,
    std::shared_ptr<ThreadSafeQueue<ProcessedResult>> results_queue,
    int num_lambdas,
    int height,
    int width
) : raw_frame_queue_(std::move(raw_frame_queue)),
    results_queue_(std::move(results_queue)),
    num_lambdas_(num_lambdas),
    height_(height),
    width_(width) {

    reinitialize(num_lambdas_, height_, width_);
}

ProcessorThread::~ProcessorThread() {
    stop();
}

void ProcessorThread::reinitialize(int num_lambdas, int height, int width) {
    num_lambdas_ = num_lambdas;
    height_ = height;
    width_ = width;

    size_t vol_size = (size_t)num_lambdas_ * height_ * width_;
    sweep_buffer_.assign(vol_size, 0.0f);
    buffer_valid_mask_.assign(num_lambdas_, 0);

    h_line_buffer_.assign((size_t)num_lambdas_ * width_, 0.0f);
    v_line_buffer_.assign((size_t)num_lambdas_ * height_, 0.0f);
    ascan_line_buffer_.assign(num_lambdas_, 0.0f);
    last_camera_frame_.assign((size_t)height_ * width_, 0.0f);

    cpu_engine_ = std::make_unique<CpuComputeEngine>(num_lambdas_);
    metal_engine_ = create_metal_compute_engine(num_lambdas_);

    crosshair_ = {width_ / 2, height_ / 2};
}

void ProcessorThread::start() {
    if (running_.exchange(true)) return;
    paused_.store(false);
    thread_ = std::thread(&ProcessorThread::thread_loop, this);
    std::cout << "[ProcessorThread] Started processing thread at full " 
              << width_ << "x" << height_ << " resolution." << std::endl;
}

void ProcessorThread::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void ProcessorThread::pause() {
    paused_.store(true);
}

void ProcessorThread::resume() {
    paused_.store(false);
}

void ProcessorThread::set_crosshair(int x, int y) {
    std::lock_guard<std::mutex> lock(crosshair_mutex_);
    crosshair_.x = std::clamp(x, 0, width_ - 1);
    crosshair_.y = std::clamp(y, 0, height_ - 1);
    crosshair_dirty_.store(true);
}

CrosshairPos ProcessorThread::get_crosshair() const {
    std::lock_guard<std::mutex> lock(crosshair_mutex_);
    return crosshair_;
}

void ProcessorThread::set_approach(ProcessingApproach approach) {
    approach_.store(approach);
}

ProcessingApproach ProcessorThread::get_approach() const {
    return approach_.load();
}

void ProcessorThread::set_backend(ComputeBackend backend) {
    backend_.store(backend);
}

ComputeBackend ProcessorThread::get_backend() const {
    return backend_.load();
}

ProcessorStats ProcessorThread::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void ProcessorThread::set_reference_dataset(std::shared_ptr<Dataset> ref_dataset) {
    ref_dataset_ = std::move(ref_dataset);
}

void ProcessorThread::set_reference_subtraction_enabled(bool enabled) {
    enable_ref_sub_.store(enabled);
}

bool ProcessorThread::get_reference_subtraction_enabled() const {
    return enable_ref_sub_.load();
}

bool ProcessorThread::has_reference_dataset() const {
    return ref_dataset_ != nullptr;
}

void ProcessorThread::set_illumination_correction_enabled(bool enabled) {
    enable_illum_corr_.store(enabled);
}

bool ProcessorThread::get_illumination_correction_enabled() const {
    return enable_illum_corr_.load();
}

void ProcessorThread::extract_crosshair_from_volume(const CrosshairPos& pos) {
    size_t frame_pixels = (size_t)height_ * width_;
    for (int l = 0; l < num_lambdas_; ++l) {
        const float* frame = sweep_buffer_.data() + (size_t)l * frame_pixels;
        // Horizontal row
        std::memcpy(h_line_buffer_.data() + (size_t)l * width_,
                    frame + (size_t)pos.y * width_,
                    width_ * sizeof(float));
        // Vertical column
        float* v_dst = v_line_buffer_.data() + (size_t)l * height_;
        for (int y = 0; y < height_; ++y) {
            v_dst[y] = frame[(size_t)y * width_ + pos.x];
        }
        // Central A-scan
        ascan_line_buffer_[l] = frame[(size_t)pos.y * width_ + pos.x];
    }
}

void ProcessorThread::thread_loop() {
    auto last_fps_time = std::chrono::steady_clock::now();
    uint64_t processed_frames = 0;

    while (running_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        RawFrameMessage msg;
        if (!raw_frame_queue_->pop(msg, 20)) {
            continue;
        }

        if (msg.is_dropped || !msg.frame_ptr) {
            continue;
        }

        int l_idx = msg.lambda_index;
        if (l_idx < 0 || l_idx >= num_lambdas_) continue;

        size_t frame_pixels = (size_t)height_ * width_;
        const float* src_ptr = msg.frame_ptr;

        // Store into resident volume buffer directly (pure raw sweeping frames)
        std::memcpy(sweep_buffer_.data() + (size_t)l_idx * frame_pixels,
                    src_ptr, frame_pixels * sizeof(float));
        buffer_valid_mask_[l_idx] = 1;
        frames_in_current_sweep_++;

        CrosshairPos pos = get_crosshair();

        // Update streaming lines
        std::memcpy(h_line_buffer_.data() + (size_t)l_idx * width_,
                    src_ptr + (size_t)pos.y * width_,
                    width_ * sizeof(float));

        float* v_dst = v_line_buffer_.data() + (size_t)l_idx * height_;
        for (int y = 0; y < height_; ++y) {
            v_dst[y] = src_ptr[(size_t)y * width_ + pos.x];
        }

        ascan_line_buffer_[l_idx] = src_ptr[(size_t)pos.y * width_ + pos.x];

        // Store latest 2D camera frame
        std::memcpy(last_camera_frame_.data(), src_ptr, frame_pixels * sizeof(float));

        if (crosshair_dirty_.exchange(false)) {
            extract_crosshair_from_volume(pos);
        }

        bool should_compute = false;
        if (approach_.load() == ProcessingApproach::RollingBuffer) {
            should_compute = true;
        } else {
            // Full Sweep: compute when sweep finishes or reaches frame count
            if (l_idx >= num_lambdas_ - 1 || frames_in_current_sweep_ >= num_lambdas_) {
                should_compute = true;
                frames_in_current_sweep_ = 0;
            }
        }

        if (should_compute) {
            auto compute_start = std::chrono::steady_clock::now();

            CrosshairReconResult recon;
            ComputeBackend backend = backend_.load();

            if (backend == ComputeBackend::MetalGPU && metal_engine_ && metal_engine_->is_available()) {
                metal_engine_->compute_from_lines(
                    h_line_buffer_.data(), v_line_buffer_.data(), ascan_line_buffer_.data(),
                    num_lambdas_, width_, height_, recon
                );
            } else {
                cpu_engine_->compute_from_lines(
                    h_line_buffer_.data(), v_line_buffer_.data(), ascan_line_buffer_.data(),
                    num_lambdas_, width_, height_, recon
                );
            }

            auto compute_end = std::chrono::steady_clock::now();
            float latency_ms = std::chrono::duration<float, std::milli>(compute_end - compute_start).count();

            ProcessedResult result;
            result.camera_frame = last_camera_frame_;
            result.h_bscan = std::move(recon.h_bscan);
            result.v_bscan = std::move(recon.v_bscan);
            result.ascan = std::move(recon.ascan);
            result.depth_axis = std::move(recon.depth_axis);
            result.crosshair = pos;
            result.sweep_index = msg.sweep_index;
            result.lambda_index = msg.lambda_index;
            result.compute_latency_ms = latency_ms;
            result.approach = approach_.load();
            result.backend = backend;

            // Compute buffer fill ratio
            int filled = 0;
            for (uint8_t m : buffer_valid_mask_) if (m) filled++;
            result.buffer_fill_ratio = (float)filled / (float)num_lambdas_;

            results_queue_->push(std::move(result));
            processed_frames++;

            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.last_compute_latency_ms = latency_ms;
            stats_.buffer_fill_ratio = result.buffer_fill_ratio;
        }

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();
        if (elapsed >= 250) {
            float fps = (float)processed_frames * 1000.0f / (float)elapsed;
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.fps = fps;
            stats_.frames_processed += processed_frames;
            processed_frames = 0;
            last_fps_time = now;
        }
    }
}

} // namespace oct

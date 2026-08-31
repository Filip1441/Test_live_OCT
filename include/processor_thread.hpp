#pragma once

#include <memory>
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>
#include "types.hpp"
#include "config.hpp"
#include "thread_safe_queue.hpp"
#include "compute_engine.hpp"

namespace oct {

class ProcessorThread {
public:
    ProcessorThread(
        std::shared_ptr<ThreadSafeQueue<RawFrameMessage>> raw_frame_queue,
        std::shared_ptr<ThreadSafeQueue<ProcessedResult>> results_queue,
        int num_lambdas = SWEEP_POINTS,
        int height = RAW_HEIGHT,
        int width = RAW_WIDTH
    );
    ~ProcessorThread();

    void start();
    void stop();
    void pause();
    void resume();
    void reinitialize(int num_lambdas, int height, int width);

    void set_crosshair(int x, int y);
    CrosshairPos get_crosshair() const;

    void set_approach(ProcessingApproach approach);
    ProcessingApproach get_approach() const;

    void set_backend(ComputeBackend backend);
    ComputeBackend get_backend() const;

    ProcessorStats get_stats() const;

    void set_reference_dataset(std::shared_ptr<Dataset> ref_dataset);
    void set_reference_subtraction_enabled(bool enabled);
    bool get_reference_subtraction_enabled() const;
    bool has_reference_dataset() const;

    void set_illumination_correction_enabled(bool enabled);
    bool get_illumination_correction_enabled() const;

    const float* get_volume_data() const { return sweep_buffer_.data(); }
    int get_num_lambdas() const { return num_lambdas_; }
    int get_height() const { return height_; }
    int get_width() const { return width_; }

    void set_wavelengths(std::vector<double> wavelengths) { wavelengths_ = std::move(wavelengths); }
    const std::vector<double>& get_wavelengths() const { return wavelengths_; }
    std::shared_ptr<Dataset> get_reference_dataset() const { return ref_dataset_; }

private:
    std::vector<double> wavelengths_;
    void thread_loop();
    void extract_crosshair_from_volume(const CrosshairPos& pos);

    std::shared_ptr<ThreadSafeQueue<RawFrameMessage>> raw_frame_queue_;
    std::shared_ptr<ThreadSafeQueue<ProcessedResult>> results_queue_;

    int num_lambdas_;
    int height_;
    int width_;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};

    std::atomic<ProcessingApproach> approach_{ProcessingApproach::RollingBuffer};
    std::atomic<ComputeBackend> backend_{ComputeBackend::MetalGPU};

    mutable std::mutex crosshair_mutex_;
    CrosshairPos crosshair_{RAW_WIDTH / 2, RAW_HEIGHT / 2};
    std::atomic<bool> crosshair_dirty_{false};

    std::shared_ptr<Dataset> ref_dataset_;
    std::atomic<bool> enable_ref_sub_{false};
    std::atomic<bool> enable_illum_corr_{true};
    std::vector<float> illum_field_;
    std::vector<float> corrected_frame_scratch_;

    mutable std::mutex stats_mutex_;
    ProcessorStats stats_;

    // Full Sweep 6.79 GB volume buffer
    std::vector<float> sweep_buffer_;
    std::vector<uint8_t> buffer_valid_mask_;
    int frames_in_current_sweep_ = 0;

    // Dedicated contiguous streaming line buffers
    std::vector<float> h_line_buffer_;
    std::vector<float> v_line_buffer_;
    std::vector<float> ascan_line_buffer_;
    std::vector<float> last_camera_frame_;

    std::unique_ptr<IComputeEngine> cpu_engine_;
    std::unique_ptr<IComputeEngine> metal_engine_;
};

} // namespace oct

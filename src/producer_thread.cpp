#include "producer_thread.hpp"
#include <chrono>
#include <random>
#include <iostream>

namespace oct {

ProducerThread::ProducerThread(
    std::shared_ptr<Dataset> dataset,
    std::shared_ptr<ThreadSafeQueue<RawFrameMessage>> queue
) : dataset_(std::move(dataset)), queue_(std::move(queue)) {}

ProducerThread::~ProducerThread() {
    stop();
}

void ProducerThread::start() {
    if (running_.exchange(true)) return;
    paused_.store(false);
    thread_ = std::thread(&ProducerThread::thread_loop, this);
}

void ProducerThread::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) {
        thread_.join();
    }
}

void ProducerThread::pause() {
    paused_.store(true);
}

void ProducerThread::resume() {
    paused_.store(false);
}

void ProducerThread::set_dataset(std::shared_ptr<Dataset> dataset) {
    std::lock_guard<std::mutex> lock(dataset_mutex_);
    dataset_ = std::move(dataset);
}

ProducerStats ProducerThread::get_stats() const {
    std::lock_guard<std::mutex> lock(stats_mutex_);
    return stats_;
}

void ProducerThread::thread_loop() {
    int current_lambda_idx = 0;
    int current_sweep = 0;

    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> rand_dist(0.0f, 1.0f);

    auto last_fps_time = std::chrono::steady_clock::now();
    uint64_t frame_counter = 0;
    uint64_t dropped_counter = 0;

    while (running_.load()) {
        if (paused_.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        std::shared_ptr<Dataset> ds;
        {
            std::lock_guard<std::mutex> lock(dataset_mutex_);
            ds = dataset_;
        }

        if (!ds || ds->num_wavelengths == 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        int num_lambdas = ds->num_wavelengths;
        float step_nm = ds->lambda_step_nm > 0.0f ? ds->lambda_step_nm : DEFAULT_LAMBDA_STEP_NM;
        float speed = tuning_speed_nm_s_.load();
        float target_fps = (speed > 0.0f) ? (speed / step_nm) : 100.0f;
        target_fps = std::clamp(target_fps, 1.0f, 50000.0f);

        auto frame_interval = std::chrono::nanoseconds(static_cast<int64_t>(1e9 / target_fps));
        auto frame_start_time = std::chrono::steady_clock::now();

        // Check for simulated frame drops
        bool is_drop = false;
        if (simulate_drops_.load()) {
            float p = rand_dist(rng);
            if (p < drop_rate_.load()) {
                is_drop = true;
                dropped_counter++;
            }
        }

        if (!is_drop) {
            RawFrameMessage msg;
            if (simulate_drift_.load()) {
                auto now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    frame_start_time.time_since_epoch()
                ).count();
                double t = (double)now_ns * 1e-9;
                int shift_x = (int)std::round(24.0 * std::sin(t * 1.8) + 10.0 * std::cos(t * 4.2));
                int shift_y = (int)std::round(18.0 * std::cos(t * 1.5) + 8.0 * std::sin(t * 3.3));

                msg.frame_data.resize((size_t)ds->width * ds->height);
                const float* src = ds->get_frame(current_lambda_idx);
                int w = ds->width, h = ds->height;
                for (int y = 0; y < h; ++y) {
                    int sy = std::clamp(y - shift_y, 0, h - 1);
                    for (int x = 0; x < w; ++x) {
                        int sx = std::clamp(x - shift_x, 0, w - 1);
                        msg.frame_data[(size_t)y * w + x] = src[(size_t)sy * w + sx];
                    }
                }
                msg.frame_ptr = msg.frame_data.data();
            } else {
                msg.frame_ptr = ds->get_frame(current_lambda_idx);
            }

            msg.lambda_index = current_lambda_idx;
            msg.sweep_index = current_sweep;
            if (!ds->wavelengths_nm.empty() && current_lambda_idx < (int)ds->wavelengths_nm.size()) {
                msg.wavelength_nm = (float)ds->wavelengths_nm[current_lambda_idx];
            } else {
                msg.wavelength_nm = 803.0f + current_lambda_idx * step_nm;
            }
            msg.timestamp_ns = (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
                frame_start_time.time_since_epoch()
            ).count();
            msg.is_dropped = false;

            queue_->push(std::move(msg));
            frame_counter++;
        }

        current_lambda_idx++;
        if (current_lambda_idx >= num_lambdas) {
            current_lambda_idx = 0;
            current_sweep++;
        }

        // Update stats every 250ms
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_fps_time).count();
        if (elapsed >= 250) {
            float fps = (float)frame_counter * 1000.0f / (float)elapsed;
            std::lock_guard<std::mutex> lock(stats_mutex_);
            stats_.fps = fps;
            stats_.frames_emitted += frame_counter;
            stats_.frames_dropped += dropped_counter;
            stats_.current_sweep = current_sweep;
            stats_.current_lambda_idx = current_lambda_idx;
            if (!ds->wavelengths_nm.empty() && current_lambda_idx < (int)ds->wavelengths_nm.size()) {
                stats_.current_wavelength_nm = (float)ds->wavelengths_nm[current_lambda_idx];
            } else {
                stats_.current_wavelength_nm = 803.0f + current_lambda_idx * step_nm;
            }
            frame_counter = 0;
            dropped_counter = 0;
            last_fps_time = now;
        }

        // Rate limiting to target_fps
        auto frame_end_time = std::chrono::steady_clock::now();
        auto compute_dur = frame_end_time - frame_start_time;
        if (compute_dur < frame_interval) {
            std::this_thread::sleep_for(frame_interval - compute_dur);
        }
    }
}

} // namespace oct

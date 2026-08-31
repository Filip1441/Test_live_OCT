#pragma once

#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include "types.hpp"
#include "config.hpp"
#include "thread_safe_queue.hpp"

namespace oct {

class ProducerThread {
public:
    ProducerThread(
        std::shared_ptr<Dataset> dataset,
        std::shared_ptr<ThreadSafeQueue<RawFrameMessage>> queue
    );
    ~ProducerThread();

    void start();
    void stop();
    void pause();
    void resume();

    void set_dataset(std::shared_ptr<Dataset> dataset);
    
    float get_tuning_speed() const { return tuning_speed_nm_s_.load(); }
    void set_tuning_speed(float speed) { tuning_speed_nm_s_.store(speed); }

    bool get_simulate_drift() const { return simulate_drift_.load(); }
    void set_simulate_drift(bool enable) { simulate_drift_.store(enable); }

    bool get_simulate_drops() const { return simulate_drops_.load(); }
    void set_simulate_drops(bool enable) { simulate_drops_.store(enable); }

    float get_drop_rate() const { return drop_rate_.load(); }
    void set_drop_rate(float rate) { drop_rate_.store(rate); }

    ProducerStats get_stats() const;

private:
    void thread_loop();

    std::shared_ptr<Dataset> dataset_;
    mutable std::mutex dataset_mutex_;

    std::shared_ptr<ThreadSafeQueue<RawFrameMessage>> queue_;

    std::thread thread_;
    std::atomic<bool> running_{false};
    std::atomic<bool> paused_{false};

    std::atomic<float> tuning_speed_nm_s_{DEFAULT_TUNING_SPEED_NM_S};
    std::atomic<bool> simulate_drift_{DEFAULT_DRIFT_ENABLED};
    std::atomic<bool> simulate_drops_{DEFAULT_DROP_SIMULATION_ENABLED};
    std::atomic<float> drop_rate_{DROP_PROBABILITY};

    mutable std::mutex stats_mutex_;
    ProducerStats stats_;
    std::vector<float> drift_buffer_;
};

} // namespace oct

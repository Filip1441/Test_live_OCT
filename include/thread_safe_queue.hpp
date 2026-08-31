#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <optional>

namespace oct {

template <typename T>
class ThreadSafeQueue {
public:
    explicit ThreadSafeQueue(size_t max_size = 1000) : max_size_(max_size) {}

    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.size() >= max_size_) {
            queue_.pop(); // Drop oldest to prevent pipeline stalls
        }
        queue_.push(std::move(item));
        cv_.notify_one();
    }

    bool pop(T& item, int timeout_ms = 50) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            if (cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms), [this] { return !queue_.empty(); })) {
                item = std::move(queue_.front());
                queue_.pop();
                return true;
            }
            return false;
        }
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    std::optional<T> try_pop_latest() {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            return std::nullopt;
        }
        T item = std::move(queue_.back());
        while (!queue_.empty()) {
            queue_.pop();
        }
        return item;
    }

    bool empty() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() const {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear() {
        std::unique_lock<std::mutex> lock(mutex_);
        while (!queue_.empty()) {
            queue_.pop();
        }
    }

private:
    std::queue<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    size_t max_size_;
};

} // namespace oct

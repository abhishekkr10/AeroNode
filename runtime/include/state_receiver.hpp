#pragma once

#include "runtime_state.hpp"

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <zmq.hpp>

namespace aeronode {

struct ReceiverStats {
  std::uint64_t messages_received = 0;
  std::uint64_t malformed_dropped = 0;
  std::uint64_t out_of_order_dropped = 0;
};

class StateReceiver {
 public:
  explicit StateReceiver(std::string endpoint = "tcp://127.0.0.1:5555",
                         std::string topic = "runtime_state",
                         int ttl_ms = 2000)
      : endpoint_(std::move(endpoint)),
        topic_(std::move(topic)),
        ttl_ms_(ttl_ms),
        latest_state_(RuntimeState::ConservativeDefault()) {}

  ~StateReceiver() {
    Stop();
  }

  void Start() {
    if (running_.exchange(true)) {
      return;  // Already running
    }
    worker_thread_ = std::thread(&StateReceiver::WorkerLoop, this);
  }

  void Stop() {
    if (!running_.exchange(false)) {
      return;  // Already stopped
    }
    if (worker_thread_.joinable()) {
      worker_thread_.join();
    }
  }

  // Thread-safe copy of latest state snapshot with TTL staleness check
  RuntimeState GetLatestSnapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    RuntimeState state = latest_state_;
    if (last_received_steady_time_.time_since_epoch().count() == 0) {
      state.is_stale = true;
      return state;
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - last_received_steady_time_)
                             .count();
    if (elapsed > ttl_ms_) {
      state = RuntimeState::ConservativeDefault();
      state.is_stale = true;
    }
    return state;
  }

  ReceiverStats GetStats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
  }

 private:
  void WorkerLoop() {
    zmq::context_t context(1);
    zmq::socket_t socket(context, zmq::socket_type::sub);
    socket.set(zmq::sockopt::rcvtimeo, 100);  // 100ms poll timeout for clean cancellation
    socket.set(zmq::sockopt::linger, 0);

    try {
      socket.connect(endpoint_);
      socket.set(zmq::sockopt::subscribe, topic_);
    } catch (...) {
      return;
    }

    std::int64_t last_seq = -1;

    while (running_.load()) {
      zmq::message_t topic_msg;
      auto res = socket.recv(topic_msg, zmq::recv_flags::none);
      if (!res.has_value()) {
        continue;  // Poll timeout, check running_
      }

      if (!socket.get(zmq::sockopt::rcvmore)) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.malformed_dropped++;
        continue;
      }

      zmq::message_t payload_msg;
      res = socket.recv(payload_msg, zmq::recv_flags::none);
      if (!res.has_value()) {
        continue;
      }

      const std::string payload(static_cast<const char*>(payload_msg.data()), payload_msg.size());
      RuntimeState parsed;
      if (!RuntimeState::ParseJson(payload, parsed)) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.malformed_dropped++;
        continue;
      }

      // Check sequence monotonicity
      if (last_seq >= 0 && parsed.sequence <= last_seq) {
        std::lock_guard<std::mutex> lock(mutex_);
        stats_.out_of_order_dropped++;
        continue;
      }
      last_seq = parsed.sequence;

      // Update state snapshot
      {
        std::lock_guard<std::mutex> lock(mutex_);
        latest_state_ = parsed;
        last_received_steady_time_ = std::chrono::steady_clock::now();
        stats_.messages_received++;
      }
    }

    try {
      socket.close();
      context.close();
    } catch (...) {
      // Ignored during shutdown
    }
  }

  std::string endpoint_;
  std::string topic_;
  int ttl_ms_;

  std::atomic<bool> running_{false};
  std::thread worker_thread_;

  mutable std::mutex mutex_;
  RuntimeState latest_state_;
  std::chrono::steady_clock::time_point last_received_steady_time_{};
  ReceiverStats stats_{};
};

}  // namespace aeronode

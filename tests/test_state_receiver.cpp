#include "state_receiver.hpp"

#include <cassert>
#include <chrono>
#include <iostream>
#include <thread>
#include <zmq.hpp>

void TestRuntimeStateParsing() {
  // 1. Valid JSON payload
  const std::string valid_json =
      R"({"schema_version":1,"sequence":42,"timestamp_unix_ms":1789500000000,"cpu_load":85.5,"temperature_c":72.0,"latency_budget_ms":20.0,"resource_pressure":0.75})";
  aeronode::RuntimeState state;
  bool ok = aeronode::RuntimeState::ParseJson(valid_json, state);
  assert(ok);
  assert(state.sequence == 42);
  assert(state.cpu_load == 85.5F);
  assert(state.temperature_c == 72.0F);
  assert(state.latency_budget_ms == 20.0F);
  assert(state.resource_pressure == 0.75F);
  assert(!state.is_stale);

  // 2. Out-of-bounds: negative latency budget
  const std::string invalid_budget =
      R"({"schema_version":1,"sequence":43,"timestamp_unix_ms":1789500000000,"cpu_load":85.5,"temperature_c":72.0,"latency_budget_ms":-5.0,"resource_pressure":0.75})";
  ok = aeronode::RuntimeState::ParseJson(invalid_budget, state);
  assert(!ok);

  // 3. Out-of-bounds: pressure > 1.0
  const std::string invalid_pressure =
      R"({"schema_version":1,"sequence":44,"timestamp_unix_ms":1789500000000,"cpu_load":85.5,"temperature_c":72.0,"latency_budget_ms":20.0,"resource_pressure":1.5})";
  ok = aeronode::RuntimeState::ParseJson(invalid_pressure, state);
  assert(!ok);

  // 4. Malformed JSON
  ok = aeronode::RuntimeState::ParseJson("not a json payload", state);
  assert(!ok);

  std::cout << "[PASS] TestRuntimeStateParsing\n";
}

void TestStateReceiverLifecycleAndFallback() {
  // Start receiver on an unused port with 200ms TTL
  aeronode::StateReceiver receiver("tcp://127.0.0.1:5591", "runtime_state", 200);
  receiver.Start();

  // Before receiving any message, latest snapshot must be conservative fallback & stale
  auto initial_state = receiver.GetLatestSnapshot();
  assert(initial_state.is_stale);
  assert(initial_state.cpu_load >= 80.0F);  // Conservative load

  receiver.Stop();
  std::cout << "[PASS] TestStateReceiverLifecycleAndFallback\n";
}

void TestStateReceiverMessageFlow() {
  const std::string endpoint = "tcp://127.0.0.1:5592";
  const std::string topic = "runtime_state";

  // Create a publisher socket in test
  zmq::context_t pub_context(1);
  zmq::socket_t pub_socket(pub_context, zmq::socket_type::pub);
  pub_socket.set(zmq::sockopt::linger, 0);
  pub_socket.bind(endpoint);

  // Start receiver with 500ms TTL
  aeronode::StateReceiver receiver(endpoint, topic, 500);
  receiver.Start();

  // Allow subscription handshake to propagate
  std::this_thread::sleep_for(std::chrono::milliseconds(200));

  // 1. Send valid message
  const std::string valid_payload =
      R"({"schema_version":1,"sequence":1,"timestamp_unix_ms":1000,"cpu_load":30.0,"temperature_c":50.0,"latency_budget_ms":40.0,"resource_pressure":0.25})";
  pub_socket.send(zmq::message_t(topic), zmq::send_flags::sndmore);
  pub_socket.send(zmq::message_t(valid_payload), zmq::send_flags::none);

  // Wait briefly for worker thread to process
  std::this_thread::sleep_for(std::chrono::milliseconds(150));

  auto snap1 = receiver.GetLatestSnapshot();
  assert(!snap1.is_stale);
  assert(snap1.sequence == 1);
  assert(snap1.cpu_load == 30.0F);

  // 2. Send malformed message (should be dropped)
  pub_socket.send(zmq::message_t(topic), zmq::send_flags::sndmore);
  pub_socket.send(zmq::message_t(std::string("bad_json")), zmq::send_flags::none);

  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  auto snap2 = receiver.GetLatestSnapshot();
  assert(snap2.sequence == 1);  // Retains last valid state
  auto stats = receiver.GetStats();
  assert(stats.malformed_dropped == 1);

  // 3. Send out-of-order sequence (sequence 0 <= 1, should be dropped)
  const std::string stale_seq_payload =
      R"({"schema_version":1,"sequence":0,"timestamp_unix_ms":900,"cpu_load":30.0,"temperature_c":50.0,"latency_budget_ms":40.0,"resource_pressure":0.25})";
  pub_socket.send(zmq::message_t(topic), zmq::send_flags::sndmore);
  pub_socket.send(zmq::message_t(stale_seq_payload), zmq::send_flags::none);

  std::this_thread::sleep_for(std::chrono::milliseconds(150));
  stats = receiver.GetStats();
  assert(stats.out_of_order_dropped == 1);

  // 4. Wait for TTL expiry (> 500ms)
  std::this_thread::sleep_for(std::chrono::milliseconds(550));
  auto snap_stale = receiver.GetLatestSnapshot();
  assert(snap_stale.is_stale);
  assert(snap_stale.cpu_load >= 80.0F);  // Must revert to conservative fallback
  assert(snap_stale.latency_budget_ms <= 15.0F);

  receiver.Stop();
  pub_socket.close();
  pub_context.close();
  std::cout << "[PASS] TestStateReceiverMessageFlow\n";
}

int main() {
  try {
    TestRuntimeStateParsing();
    TestStateReceiverLifecycleAndFallback();
    TestStateReceiverMessageFlow();
    std::cout << "\nAll 3 StateReceiver tests passed successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "StateReceiver test failed: " << ex.what() << '\n';
    return 1;
  }
}

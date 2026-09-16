#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <nlohmann/json.hpp>

namespace aeronode {

struct RuntimeState {
  int schema_version = 1;
  std::int64_t sequence = 0;
  std::int64_t timestamp_unix_ms = 0;
  float cpu_load = 50.0F;            // 0.0 to 100.0
  float temperature_c = 60.0F;       // degrees Celsius
  float latency_budget_ms = 30.0F;   // positive (> 0.0)
  float resource_pressure = 0.50F;   // 0.0 to 1.0
  bool is_stale = false;             // True if older than TTL or uninitialized

  bool IsValid() const {
    return schema_version == 1 &&
           sequence >= 0 &&
           !std::isnan(cpu_load) && !std::isinf(cpu_load) &&
           cpu_load >= 0.0F && cpu_load <= 100.0F &&
           !std::isnan(temperature_c) && !std::isinf(temperature_c) &&
           !std::isnan(latency_budget_ms) && !std::isinf(latency_budget_ms) &&
           latency_budget_ms > 0.0F &&
           !std::isnan(resource_pressure) && !std::isinf(resource_pressure) &&
           resource_pressure >= 0.0F && resource_pressure <= 1.0F;
  }

  // Safe conservative fallback state
  static RuntimeState ConservativeDefault() {
    RuntimeState state;
    state.schema_version = 1;
    state.sequence = 0;
    state.timestamp_unix_ms = 0;
    state.cpu_load = 85.0F;
    state.temperature_c = 85.0F;
    state.latency_budget_ms = 15.0F;
    state.resource_pressure = 0.85F;
    state.is_stale = true;
    return state;
  }

  static bool ParseJson(const std::string& payload, RuntimeState& out_state) {
    try {
      const auto j = nlohmann::json::parse(payload);
      RuntimeState state;
      state.schema_version = j.at("schema_version").get<int>();
      state.sequence = j.at("sequence").get<std::int64_t>();
      state.timestamp_unix_ms = j.at("timestamp_unix_ms").get<std::int64_t>();
      state.cpu_load = j.at("cpu_load").get<float>();
      state.temperature_c = j.at("temperature_c").get<float>();
      state.latency_budget_ms = j.at("latency_budget_ms").get<float>();
      state.resource_pressure = j.at("resource_pressure").get<float>();
      state.is_stale = false;

      if (!state.IsValid()) {
        return false;
      }
      out_state = state;
      return true;
    } catch (...) {
      return false;
    }
  }
};

}  // namespace aeronode

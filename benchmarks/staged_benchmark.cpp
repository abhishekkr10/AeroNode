// Phase 5E: persistent native execution with paired, untimed input/output I/O.
#include "staged_model_runner.hpp"
#include "policy_config.hpp"
#include "state_receiver.hpp"
#include <memory>
#include <thread>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#endif

namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t input_elements = 3 * 128 * 128;
std::vector<std::string> modes = {
    "monolithic_full", "staged_exit_1", "staged_exit_2", "staged_exit_3", "staged_exit_4", "adaptive_current"};

struct Options {
  std::string input, labels, stage_dir, model, records;
  std::string policy_path, ipc_endpoint;
  bool cpu_arena = true, memory_pattern = true;
  bool include_static = false;
  aeronode::RuntimeState adaptive_state;
  int samples = 0, repeats = 3, warmup = 20, seed = 20261012;
};

int PositiveInt(const std::string& value) {
  std::size_t end = 0;
  const auto number = std::stoll(value, &end);
  if (end != value.size() || number <= 0 || number > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("Expected positive 32-bit integer, got " + value);
  }
  return static_cast<int>(number);
}

Options Parse(int argc, char** argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string key = argv[i];
    if (++i == argc) throw std::invalid_argument("Missing value for " + key);
    const std::string value = argv[i];
    if (key == "--input-bin") options.input = value;
    else if (key == "--labels-bin") options.labels = value;
    else if (key == "--stage-dir") options.stage_dir = value;
    else if (key == "--monolithic-model") options.model = value;
    else if (key == "--records") options.records = value;
    else if (key == "--policy-config") options.policy_path = value;
    else if (key == "--ipc-endpoint") options.ipc_endpoint = value;
    else if (key == "--include-static") {
      if (value != "0" && value != "1") throw std::invalid_argument("Static switch requires 0 or 1");
      options.include_static = value == "1";
    }
    else if (key == "--latency-budget" || key == "--resource-pressure" || key == "--cpu-load" || key == "--temperature") {
      std::size_t end = 0;
      const float number = std::stof(value,&end);
      if (end != value.size() || !std::isfinite(number)) throw std::invalid_argument("RuntimeState value must be finite numeric");
      if (key == "--latency-budget") options.adaptive_state.latency_budget_ms = number;
      if (key == "--resource-pressure") options.adaptive_state.resource_pressure = number;
      if (key == "--cpu-load") options.adaptive_state.cpu_load = number;
      if (key == "--temperature") options.adaptive_state.temperature_c = number;
    }
    else if (key == "--cpu-arena" || key == "--memory-pattern") {
      if (value != "0" && value != "1") throw std::invalid_argument("Allocator switches require 0 or 1");
      if (key == "--cpu-arena") options.cpu_arena = value == "1";
      else options.memory_pattern = value == "1";
    }
    else if (key == "--samples") options.samples = PositiveInt(value);
    else if (key == "--repeats") options.repeats = PositiveInt(value);
    else if (key == "--warmup") options.warmup = PositiveInt(value);
    else if (key == "--seed") options.seed = PositiveInt(value);
    else throw std::invalid_argument("Unknown benchmark option " + key);
  }
  if (options.samples == 0 || options.input.empty() || options.labels.empty() ||
      options.stage_dir.empty() || options.model.empty() || options.records.empty()) {
    throw std::invalid_argument("Required: --input-bin --labels-bin --stage-dir --monolithic-model --records --samples");
  }
  if (!options.adaptive_state.IsValid()) throw std::invalid_argument("Invalid controlled RuntimeState");
  return options;
}

double CpuMilliseconds() {
#ifdef _WIN32
  FILETIME created{}, exited{}, kernel{}, user{};
  if (!GetProcessTimes(GetCurrentProcess(), &created, &exited, &kernel, &user)) {
    throw std::runtime_error("GetProcessTimes failed");
  }
  const auto ticks = [](FILETIME time) {
    return (static_cast<std::uint64_t>(time.dwHighDateTime) << 32) | time.dwLowDateTime;
  };
  return static_cast<double>(ticks(kernel) + ticks(user)) / 10000.0;
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
  return (usage.ru_utime.tv_sec + usage.ru_stime.tv_sec) * 1000.0 +
         (usage.ru_utime.tv_usec + usage.ru_stime.tv_usec) / 1000.0;
#endif
}

std::uint64_t PeakWorkingSet() {
#ifdef _WIN32
  PROCESS_MEMORY_COUNTERS counters{};
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &counters, sizeof(counters))) {
    throw std::runtime_error("GetProcessMemoryInfo failed");
  }
  return static_cast<std::uint64_t>(counters.PeakWorkingSetSize);
#else
  rusage usage{};
  if (getrusage(RUSAGE_SELF, &usage) != 0) throw std::runtime_error("getrusage failed");
#ifdef __APPLE__
  return static_cast<std::uint64_t>(usage.ru_maxrss);
#else
  return static_cast<std::uint64_t>(usage.ru_maxrss) * 1024;
#endif
#endif
}

void ValidateFile(const std::string& path, std::uintmax_t bytes) {
  if (!std::filesystem::is_regular_file(std::filesystem::u8path(path)) ||
      std::filesystem::file_size(std::filesystem::u8path(path)) != bytes) {
    throw std::invalid_argument("Binary file size mismatch or missing file: " + path);
  }
}

class Reference {
 public:
  explicit Reference(const std::string& path, bool cpu_arena, bool memory_pattern)
      : environment_(ORT_LOGGING_LEVEL_WARNING, "phase5e_reference"), session_(nullptr),
        memory_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    options_.SetIntraOpNumThreads(1);
    options_.SetInterOpNumThreads(1);
    if (!cpu_arena) options_.DisableCpuMemArena();
    if (!memory_pattern) options_.DisableMemPattern();
    const auto native = std::filesystem::u8path(path);
    session_ = Ort::Session(environment_, native.c_str(), options_);
    Ort::AllocatorWithDefaultOptions allocator;
    if (session_.GetInputCount() != 1 || session_.GetOutputCount() != 4) {
      throw std::runtime_error("Monolithic reference requires one input and four outputs");
    }
    const auto owner = session_.GetInputTypeInfo(0);
    if (std::string(session_.GetInputNameAllocated(0, allocator).get()) != "input" ||
        owner.GetTensorTypeAndShapeInfo().GetShape() != std::vector<std::int64_t>{-1, 3, 128, 128} ||
        owner.GetTensorTypeAndShapeInfo().GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("Invalid monolithic reference input contract");
    }
    for (std::size_t i = 0; i < 4; ++i) {
      if (std::string(session_.GetOutputNameAllocated(i, allocator).get()) != output_names_[i]) {
        throw std::runtime_error("Invalid monolithic reference output name");
      }
    }
  }

  std::array<std::vector<float>, 4> Run(std::vector<float>& input) {
    const std::array<std::int64_t, 4> shape = {1, 3, 128, 128};
    auto tensor = Ort::Value::CreateTensor<float>(memory_, input.data(), input.size(), shape.data(), shape.size());
    const char* name = "input";
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, &name, &tensor, 1, output_names_.data(), 4);
    std::array<std::vector<float>, 4> logits;
    for (std::size_t i = 0; i < 4; ++i) {
      const auto info = outputs.at(i).GetTensorTypeAndShapeInfo();
      if (info.GetShape() != std::vector<std::int64_t>{1, 10} || info.GetElementCount() != 10 ||
          info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error("Invalid monolithic reference output shape/count/type");
      }
      const auto* data = outputs[i].GetTensorData<float>();
      for (int c = 0; c < 10; ++c) {
        if (!std::isfinite(data[c])) throw std::runtime_error("Non-finite reference logit");
      }
      logits[i].assign(data, data + 10);
      // Include all-head metrics computation in the full-runtime baseline.
      aeronode::ComputePredictionMetrics(logits[i]);
    }
    return logits;
  }

 private:
  Ort::Env environment_;
  Ort::SessionOptions options_;
  Ort::Session session_;
  Ort::MemoryInfo memory_;
  std::array<const char*, 4> output_names_ = {"exit_1_logits", "exit_2_logits", "exit_3_logits", "exit_4_logits"};
};

struct Measurement {
  std::vector<float> logits;
  std::array<std::vector<float>, 4> reference;
  std::array<std::size_t, 4> stage_runs{};
  int exit = 4;
  std::string reason = "FULL_DEPTH_BASELINE";
  double latency_ms = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
  try {
    const auto options = Parse(argc, argv);
    std::unique_ptr<aeronode::StateReceiver> receiver;
    if (!options.ipc_endpoint.empty()) {
      receiver = std::make_unique<aeronode::StateReceiver>(options.ipc_endpoint);
      receiver->Start();
      const auto limit = Clock::now() + std::chrono::seconds(3);
      while (receiver->GetLatestSnapshot().is_stale && Clock::now() < limit) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      if (receiver->GetLatestSnapshot().is_stale) throw std::runtime_error("No fresh benchmark IPC state");
    }
    const auto snapshot = [&]() {
      const auto state = receiver ? receiver->GetLatestSnapshot() : options.adaptive_state;
      if (receiver && state.is_stale) throw std::runtime_error("Benchmark IPC state became stale");
      return state;
    };
    ValidateFile(options.input, static_cast<std::uintmax_t>(options.samples) * input_elements * sizeof(float));
    ValidateFile(options.labels, static_cast<std::uintmax_t>(options.samples) * sizeof(std::int32_t));
    if (std::filesystem::exists(std::filesystem::u8path(options.records))) {
      throw std::invalid_argument("Refusing to overwrite existing benchmark records");
    }
    std::ifstream images(std::filesystem::u8path(options.input), std::ios::binary);
    std::ifstream label_file(std::filesystem::u8path(options.labels), std::ios::binary);
    std::vector<std::int32_t> labels(static_cast<std::size_t>(options.samples));
    label_file.read(reinterpret_cast<char*>(labels.data()), static_cast<std::streamsize>(labels.size() * sizeof(std::int32_t)));
    if (!label_file) throw std::runtime_error("Could not read labels");
    for (const auto label : labels) if (label < 0 || label >= 10) throw std::invalid_argument("Label outside [0,9]");

    aeronode::StagedEngineConfig config;
    config.enable_cpu_mem_arena = options.cpu_arena;
    config.enable_memory_pattern = options.memory_pattern;
    for (std::size_t i = 0; i < 4; ++i) {
      config.stage_model_paths[i] = (std::filesystem::u8path(options.stage_dir) /
          ("stage_" + std::to_string(i + 1) + ".onnx")).u8string();
    }
    aeronode::StagedInferenceEngine engine(config);
    Reference reference(options.model, options.cpu_arena, options.memory_pattern);
    std::vector<aeronode::ExitDecisionEngine> policies;
    for (int exit = 1; exit <= 4; ++exit) {
      aeronode::ExitPolicyConfig policy;
      policy.entropy_thresholds = {-1.0F, -1.0F, -1.0F, -1.0F};
      if (exit < 4) policy.entropy_thresholds[static_cast<std::size_t>(exit - 1)] = 3.0F;
      policies.emplace_back(policy);
    }
    const auto adaptive_config = options.policy_path.empty() ? aeronode::ExitPolicyConfig{} :
        aeronode::LoadExitPolicy(options.policy_path);
    policies.emplace_back(adaptive_config);
    if (!options.policy_path.empty()) modes[5] = "adaptive_calibrated";
    if (options.include_static) modes.push_back(options.policy_path.empty() ? "static_current" : "static_calibrated");
    aeronode::RuntimeState baseline_state;
    baseline_state.latency_budget_ms = 1000000.0F;
    baseline_state.resource_pressure = 0.0F;
    auto adaptive_state = snapshot();
    std::vector<float> input(input_elements);
    const auto load = [&](int sample) {
      images.seekg(static_cast<std::streamoff>(sample) * static_cast<std::streamoff>(input_elements * sizeof(float)));
      images.read(reinterpret_cast<char*>(input.data()), static_cast<std::streamsize>(input_elements * sizeof(float)));
      if (!images) throw std::runtime_error("Failed to read image sample " + std::to_string(sample));
    };
    const auto run = [&](std::size_t mode, const aeronode::RuntimeState& request_state) {
      Measurement measurement;
      const auto start = Clock::now();
      if (mode == 0) {
        measurement.reference = reference.Run(input);
        measurement.logits = measurement.reference[3];
      } else {
        const auto result = engine.Infer(input, {1, 3, 128, 128},
            mode == 5 ? request_state : baseline_state, policies[mode == 6 ? 4 : mode - 1]);
        measurement.logits = result.logits;
        measurement.stage_runs = result.stage_run_counts;
        measurement.exit = result.selected_exit;
        measurement.reason = aeronode::ToString(result.stages.back().sample_evaluations[0].reason);
      }
      measurement.latency_ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      return measurement;
    };
    engine.WarmUp();
    for (int i = 0; i < options.warmup; ++i) {
      load(i % options.samples);
      const auto warmup_state = snapshot();
      for (std::size_t mode = 0; mode < modes.size(); ++mode) run(mode, warmup_state);
    }
    std::ofstream records(std::filesystem::u8path(options.records));
    if (!records) throw std::runtime_error("Cannot create benchmark records");
    const double cpu_start = CpuMilliseconds();
    const auto wall_start = Clock::now();
    std::vector<int> order(static_cast<std::size_t>(options.samples));
    for (int repetition = 0; repetition < options.repeats; ++repetition) {
      std::iota(order.begin(), order.end(), 0);
      std::mt19937 generator(static_cast<unsigned>(options.seed) + static_cast<unsigned>(repetition));
      std::shuffle(order.begin(), order.end(), generator);
      for (std::size_t position = 0; position < order.size(); ++position) {
        const int sample = order[position];
        load(sample);  // Disk I/O is outside every request timer.
        // One shared resource observation per paired sample; snapshot acquisition
        // is outside request timers, with no per-layer IPC or sensor reads.
        adaptive_state = snapshot();
        std::vector<Measurement> results(modes.size());
        for (std::size_t slot = 0; slot < modes.size(); ++slot) {
          const auto mode = (slot + position + static_cast<std::size_t>(repetition)) % modes.size();
          results[mode] = run(mode, adaptive_state);
        }
        for (std::size_t mode = 0; mode < modes.size(); ++mode) {
          const auto& result = results[mode];
          if (result.exit < 1 || result.exit > 4 || result.logits.size() != 10) {
            throw std::runtime_error("Invalid benchmark result");
          }
          const auto& expected = results[0].reference[static_cast<std::size_t>(result.exit - 1)];
          double max_error = 0.0;
          for (std::size_t c = 0; c < 10; ++c) max_error = std::max(max_error, std::abs(static_cast<double>(result.logits[c] - expected[c])));
          const auto metrics = aeronode::ComputePredictionMetrics(result.logits);
          const auto expected_metrics = aeronode::ComputePredictionMetrics(expected);
          if (max_error > 1e-5 || metrics.predicted_class != expected_metrics.predicted_class) {
            throw std::runtime_error("Parity failed during benchmark at sample " + std::to_string(sample));
          }
          const double policy_budget = mode == 5 ? adaptive_state.latency_budget_ms : baseline_state.latency_budget_ms;
          const double budget = options.include_static ? adaptive_state.latency_budget_ms : policy_budget;
          records << nlohmann::json{{"mode", modes[mode]}, {"repetition", repetition}, {"sample_index", sample},
              {"label", labels[static_cast<std::size_t>(sample)]}, {"predicted_class", metrics.predicted_class},
              {"selected_exit", result.exit}, {"stage_run_counts", result.stage_runs},
              {"monolithic_session_runs", mode == 0 ? 1 : 0}, {"latency_ms", result.latency_ms},
              {"confidence", metrics.confidence}, {"entropy", metrics.entropy}, {"reason", result.reason},
              {"latency_budget_ms", budget}, {"deadline_missed", result.latency_ms > budget},
              {"policy_latency_budget_ms", policy_budget},
              {"state_source", receiver ? "external_ipc" : "controlled"},
              {"runtime_state", {{"sequence", adaptive_state.sequence},
                {"timestamp_unix_ms", adaptive_state.timestamp_unix_ms},
                {"cpu_load", adaptive_state.cpu_load}, {"temperature_c", adaptive_state.temperature_c},
                {"resource_pressure", adaptive_state.resource_pressure}, {"is_stale", adaptive_state.is_stale}}},
              {"reference_max_abs_error", max_error}}.dump() << '\n';
        }
      }
      std::cerr << "Completed repetition " << repetition + 1 << '/' << options.repeats << '\n';
    }
    records.flush();
    if (!records) throw std::runtime_error("Failed writing benchmark records");
    const double wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - wall_start).count();
    const double cpu_ms = CpuMilliseconds() - cpu_start;
    std::cout << nlohmann::json{{"completed", true}, {"samples", options.samples}, {"repeats", options.repeats},
        {"warmup_requests_per_mode", options.warmup}, {"seed", options.seed}, {"modes", modes},
        {"batch_size", 1}, {"intra_op_threads", 1}, {"inter_op_threads", 1},
        {"onnxruntime_version", OrtGetApiBase()->GetVersionString()}, {"process_cpu_ms", cpu_ms},
        {"measurement_loop_wall_ms", wall_ms}, {"process_cpu_percent_one_core", cpu_ms / wall_ms * 100.0},
        {"process_peak_working_set_bytes", PeakWorkingSet()},
        {"adaptive_state", {{"cpu_load", adaptive_state.cpu_load}, {"temperature_c", adaptive_state.temperature_c},
          {"resource_pressure", adaptive_state.resource_pressure}, {"latency_budget_ms", adaptive_state.latency_budget_ms}}},
        {"adaptive_thresholds", adaptive_config.entropy_thresholds},
        {"cpu_mem_arena_enabled", options.cpu_arena}, {"memory_pattern_enabled", options.memory_pattern}}.dump(2) << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "aeronode_benchmark: " << error.what() << '\n';
    return 1;
  }
}

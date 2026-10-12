#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "entropy.hpp"
#include "exit_decision.hpp"
#include "runtime_state.hpp"
#include "state_receiver.hpp"
#include "staged_model_runner.hpp"

namespace {

struct Options {
  std::string model_path;
  std::string staged_dir;
  std::string input_bin_path;
  std::string ipc_endpoint;
  int iterations = 100;
  int batch_size = 1;
  bool iterations_supplied = false;

  // Manual override options (used if no live IPC endpoint is specified)
  bool has_manual_overrides = false;
  float cpu_load = 50.0F;
  float temperature_c = 60.0F;
  float latency_budget_ms = 30.0F;
  float resource_pressure = 0.50F;

  // Policy thresholds
  std::array<float, 4> entropy_thresholds = {0.60F, 0.80F, 1.10F, 2.50F};
};

std::array<float, 4> ParseThresholds(const std::string& csv) {
  std::array<float, 4> thresholds = {0.60F, 0.80F, 1.10F, 2.50F};
  std::stringstream ss(csv);
  std::string token;
  int idx = 0;
  while (std::getline(ss, token, ',') && idx < 4) {
    if (!token.empty()) {
      thresholds[idx++] = std::stof(token);
    }
  }
  return thresholds;
}

Options ParseOptions(int argc, char* argv[]) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--model" && index + 1 < argc) {
      options.model_path = argv[++index];
    } else if (argument == "--staged-dir" && index + 1 < argc) {
      options.staged_dir = argv[++index];
    } else if (argument == "--iterations" && index + 1 < argc) {
      options.iterations = std::stoi(argv[++index]);
      options.iterations_supplied = true;
    } else if (argument == "--input-bin" && index + 1 < argc) {
      options.input_bin_path = argv[++index];
    } else if (argument == "--batch-size" && index + 1 < argc) {
      options.batch_size = std::stoi(argv[++index]);
    } else if (argument == "--ipc-endpoint" && index + 1 < argc) {
      options.ipc_endpoint = argv[++index];
    } else if (argument == "--cpu-load" && index + 1 < argc) {
      options.cpu_load = std::stof(argv[++index]);
      options.has_manual_overrides = true;
    } else if (argument == "--temperature" && index + 1 < argc) {
      options.temperature_c = std::stof(argv[++index]);
      options.has_manual_overrides = true;
    } else if (argument == "--latency-budget" && index + 1 < argc) {
      options.latency_budget_ms = std::stof(argv[++index]);
      options.has_manual_overrides = true;
    } else if (argument == "--resource-pressure" && index + 1 < argc) {
      options.resource_pressure = std::stof(argv[++index]);
      options.has_manual_overrides = true;
    } else if (argument == "--entropy-thresholds" && index + 1 < argc) {
      options.entropy_thresholds = ParseThresholds(argv[++index]);
    } else {
      throw std::invalid_argument(
          "Usage: aeronode_runtime (--model <path> | --staged-dir <path>) [--iterations N] [--batch-size N] "
          "[--input-bin <path>] [--ipc-endpoint <ep>] [--cpu-load N] [--temperature N] "
          "[--latency-budget N] [--resource-pressure N] [--entropy-thresholds t1,t2,t3,t4]");
    }
  }
  if (options.model_path.empty() == options.staged_dir.empty() || options.iterations <= 0 || options.batch_size <= 0) {
    throw std::invalid_argument(
        "Usage: aeronode_runtime (--model <path> | --staged-dir <path>) [--iterations N] [--batch-size N] ...");
  }
  if (!options.staged_dir.empty() && options.iterations_supplied && options.iterations != 1) {
    throw std::invalid_argument("Phase 5D staged mode executes one request; --iterations must be 1");
  }
  return options;
}

double Percentile95(std::vector<double> samples) {
  if (samples.empty()) {
    return 0.0;
  }
  std::sort(samples.begin(), samples.end());
  const auto index = static_cast<std::size_t>(0.95 * static_cast<double>(samples.size() - 1));
  return samples.at(index);
}

std::string JsonEscape(const std::string& value) {
  std::string escaped;
  escaped.reserve(value.size());
  for (const char character : value) {
    if (character == '\\' || character == '"') {
      escaped.push_back('\\');
    }
    escaped.push_back(character);
  }
  return escaped;
}

std::vector<float> LoadBinaryFloats(const std::string& path, std::size_t expected_elements) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open binary input file: " + path);
  }
  const auto file_size = static_cast<std::size_t>(file.tellg());
  const auto expected_bytes = expected_elements * sizeof(float);
  if (file_size != expected_bytes) {
    throw std::runtime_error("Binary input file size mismatch: expected " +
                             std::to_string(expected_bytes) + " bytes, got " +
                             std::to_string(file_size) + " bytes");
  }
  file.seekg(0, std::ios::beg);
  std::vector<float> data(expected_elements);
  file.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(expected_bytes));
  if (!file) {
    throw std::runtime_error("Failed reading binary input file: " + path);
  }
  return data;
}

aeronode::RuntimeState GetStateSnapshot(const Options& options, aeronode::StateReceiver* receiver) {
  if (receiver != nullptr) return receiver->GetLatestSnapshot();
  if (!options.has_manual_overrides) return aeronode::RuntimeState::ConservativeDefault();
  aeronode::RuntimeState state;
  state.sequence = 1;
  state.timestamp_unix_ms = static_cast<std::int64_t>(std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::system_clock::now().time_since_epoch()).count());
  state.cpu_load = options.cpu_load;
  state.temperature_c = options.temperature_c;
  state.latency_budget_ms = options.latency_budget_ms;
  state.resource_pressure = options.resource_pressure;
  return state;
}

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Options options = ParseOptions(argc, argv);

    // Initialize IPC StateReceiver if an endpoint is provided
    std::unique_ptr<aeronode::StateReceiver> state_receiver;
    if (!options.ipc_endpoint.empty()) {
      state_receiver = std::make_unique<aeronode::StateReceiver>(options.ipc_endpoint);
      state_receiver->Start();
      // Allow connection & initial state delivery to arrive
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    if (!options.staged_dir.empty()) {
      aeronode::StagedEngineConfig config;
      for (std::size_t i = 0; i < 4; ++i) {
        config.stage_model_paths[i] = (std::filesystem::u8path(options.staged_dir) /
            ("stage_" + std::to_string(i + 1) + ".onnx")).u8string();
      }
      aeronode::StagedInferenceEngine engine(config);
      const auto count = static_cast<std::size_t>(options.batch_size) * 3 * 128 * 128;
      if (options.input_bin_path.empty()) {
        throw std::invalid_argument("Staged mode requires --input-bin containing preprocessed float32 images");
      }
      const auto input = LoadBinaryFloats(options.input_bin_path, count);
      const auto state = GetStateSnapshot(options, state_receiver.get());
      aeronode::ExitPolicyConfig policy_config;
      policy_config.entropy_thresholds = options.entropy_thresholds;
      const auto result = engine.Infer(input, {options.batch_size, 3, 128, 128}, state,
                                        aeronode::ExitDecisionEngine(policy_config));
      nlohmann::json report = {{"mode", "staged"}, {"selected_exit", result.selected_exit},
          {"shape", result.output_shape}, {"logits", result.logits},
          {"stage_run_counts", result.stage_run_counts}, {"stages", nlohmann::json::array()},
          {"runtime_state", {{"schema_version", state.schema_version}, {"sequence", state.sequence},
            {"timestamp_unix_ms", state.timestamp_unix_ms}, {"is_stale", state.is_stale},
            {"cpu_load", state.cpu_load}, {"temperature_c", state.temperature_c},
            {"latency_budget_ms", state.latency_budget_ms}, {"resource_pressure", state.resource_pressure}}}};
      for (const auto& stage : result.stages) {
        nlohmann::json evaluations = nlohmann::json::array();
        for (const auto& evaluation : stage.sample_evaluations) {
          evaluations.push_back({{"exit_index", evaluation.exit_index}, {"action", evaluation.action},
              {"reason", aeronode::ToString(evaluation.reason)}, {"entropy", evaluation.entropy},
              {"confidence", evaluation.confidence}, {"predicted_class", evaluation.predicted_class},
              {"threshold_used", evaluation.threshold_used}});
        }
        report["stages"].push_back({{"stage_index", stage.stage_index}, {"input_shape", stage.input_shape},
            {"output_shapes", stage.output_shapes}, {"copied_feature_elements", stage.copied_feature_elements},
            {"sample_evaluations", evaluations}});
      }
      std::cout << report.dump(2) << '\n';
      if (state_receiver) state_receiver->Stop();
      return EXIT_SUCCESS;
    }

    Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "aeronode");
    Ort::SessionOptions session_options;
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_options.SetIntraOpNumThreads(1);

#ifdef _WIN32
    const std::wstring wide_model_path(options.model_path.begin(), options.model_path.end());
    Ort::Session session(environment, wide_model_path.c_str(), session_options);
#else
    Ort::Session session(environment, options.model_path.c_str(), session_options);
#endif

    Ort::AllocatorWithDefaultOptions allocator;
    const std::size_t input_count = session.GetInputCount();
    const std::size_t output_count = session.GetOutputCount();

    if (input_count != 1) {
      throw std::runtime_error("Expected exactly 1 model input, found " + std::to_string(input_count));
    }

    const auto input_name_ptr = session.GetInputNameAllocated(0, allocator);
    const std::string input_name = input_name_ptr.get();
    const auto input_type_owner = session.GetInputTypeInfo(0);
    const auto input_info = input_type_owner.GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> raw_input_shape = input_info.GetShape();

    // Collect output names and shapes
    std::vector<Ort::AllocatedStringPtr> output_name_ptrs;
    output_name_ptrs.reserve(output_count);
    std::vector<const char*> output_names;
    output_names.reserve(output_count);
    std::vector<std::vector<std::int64_t>> output_shapes;
    output_shapes.reserve(output_count);

    for (std::size_t i = 0; i < output_count; ++i) {
      output_name_ptrs.push_back(session.GetOutputNameAllocated(i, allocator));
      output_names.push_back(output_name_ptrs.back().get());
      output_shapes.push_back(session.GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    }

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // Branch 1: Phase 1 single-output synthetic model ([1, 4] -> [1, 3])
    if (output_count == 1 && raw_input_shape.size() == 2 && raw_input_shape[1] == 4) {
      const std::vector<std::int64_t> input_shape = {1, 4};
      if (output_shapes[0] != std::vector<std::int64_t>({1, 3})) {
        throw std::runtime_error("Unexpected Phase-1 output shape; expected [1, 3]");
      }

      std::array<float, 4> input_data{-1.0F, 0.5F, 2.0F, -0.25F};
      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

      const char* input_names_arr[] = {input_name.c_str()};

      auto run_once = [&]() {
        return session.Run(Ort::RunOptions{nullptr}, input_names_arr, &input_tensor, 1,
                           output_names.data(), 1);
      };

      run_once();  // Warm-up
      std::vector<double> latency_ms;
      latency_ms.reserve(static_cast<std::size_t>(options.iterations));
      std::vector<Ort::Value> outputs;
      for (int iteration = 0; iteration < options.iterations; ++iteration) {
        const auto start = std::chrono::steady_clock::now();
        outputs = run_once();
        const auto stop = std::chrono::steady_clock::now();
        latency_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
      }

      const float* logits = outputs.at(0).GetTensorData<float>();
      const auto predicted_class =
          static_cast<int>(std::distance(logits, std::max_element(logits, logits + 3)));
      const double average_ms =
          std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0) / latency_ms.size();

      std::cout << std::fixed << std::setprecision(7)
                << "{\n"
                << "  \"model\": \"" << JsonEscape(options.model_path) << "\",\n"
                << "  \"logits\": [" << logits[0] << ", " << logits[1] << ", " << logits[2] << "],\n"
                << "  \"predicted_class\": " << predicted_class << ",\n"
                << "  \"iterations\": " << options.iterations << ",\n"
                << "  \"average_latency_ms\": " << average_ms << ",\n"
                << "  \"p95_latency_ms\": " << Percentile95(latency_ms) << "\n"
                << "}\n";

      if (state_receiver) {
        state_receiver->Stop();
      }
      return EXIT_SUCCESS;
    }

    // Branch 2: Phase 4 Multi-Exit 4-output model ([batch, 3, 128, 128] -> 4 * [batch, 10])
    if (output_count != 4) {
      throw std::runtime_error("Expected 4 outputs for multi-exit model, found " +
                               std::to_string(output_count));
    }

    const std::vector<std::string> expected_names = {
        "exit_1_logits", "exit_2_logits", "exit_3_logits", "exit_4_logits"};
    for (std::size_t i = 0; i < 4; ++i) {
      if (std::string(output_names[i]) != expected_names[i]) {
        throw std::runtime_error("Output name mismatch at index " + std::to_string(i) +
                                 ": expected " + expected_names[i] + ", got " + output_names[i]);
      }
    }

    const std::int64_t batch_size = options.batch_size;
    const std::vector<std::int64_t> input_shape = {batch_size, 3, 128, 128};
    const std::size_t total_elements = static_cast<std::size_t>(batch_size * 3 * 128 * 128);

    std::vector<float> input_data;
    if (!options.input_bin_path.empty()) {
      input_data = LoadBinaryFloats(options.input_bin_path, total_elements);
    } else {
      input_data.resize(total_elements);
      for (std::size_t i = 0; i < total_elements; ++i) {
        input_data[i] = static_cast<float>((static_cast<double>(i % 1000) - 500.0) / 500.0);
      }
    }

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());

    const char* input_names_arr[] = {input_name.c_str()};

    auto run_multi_exit = [&]() {
      return session.Run(Ort::RunOptions{nullptr}, input_names_arr, &input_tensor, 1,
                         output_names.data(), 4);
    };

    run_multi_exit();  // Warm-up iteration

    // Acquire RuntimeState snapshot: consistent snapshot used for request decisions
    const auto state_snapshot = GetStateSnapshot(options, state_receiver.get());

    std::vector<double> latency_ms;
    latency_ms.reserve(static_cast<std::size_t>(options.iterations));
    std::vector<Ort::Value> outputs;
    for (int iteration = 0; iteration < options.iterations; ++iteration) {
      const auto start = std::chrono::steady_clock::now();
      outputs = run_multi_exit();
      const auto stop = std::chrono::steady_clock::now();
      latency_ms.push_back(std::chrono::duration<double, std::milli>(stop - start).count());
    }

    const double average_ms =
        std::accumulate(latency_ms.begin(), latency_ms.end(), 0.0) / latency_ms.size();
    const double p95_ms = Percentile95(latency_ms);

    // Build policy engine
    aeronode::ExitPolicyConfig policy_config;
    policy_config.entropy_thresholds = options.entropy_thresholds;
    aeronode::ExitDecisionEngine decision_engine(policy_config);

    // Compute metrics for sample 0 across all 4 exits for early-exit decision evaluation
    std::vector<aeronode::PredictionMetrics> sample0_metrics;
    sample0_metrics.reserve(4);
    for (std::size_t exit_idx = 0; exit_idx < 4; ++exit_idx) {
      const float* logits = outputs.at(exit_idx).GetTensorData<float>();
      sample0_metrics.push_back(aeronode::ComputePredictionMetrics(logits, 10));
    }

    const auto decision = decision_engine.Evaluate(sample0_metrics, state_snapshot, average_ms);

    // Output machine-readable JSON
    std::cout << std::fixed << std::setprecision(7);
    std::cout << "{\n";
    std::cout << "  \"model\": \"" << JsonEscape(options.model_path) << "\",\n";
    std::cout << "  \"batch_size\": " << batch_size << ",\n";
    std::cout << "  \"iterations\": " << options.iterations << ",\n";
    std::cout << "  \"average_latency_ms\": " << average_ms << ",\n";
    std::cout << "  \"p95_latency_ms\": " << p95_ms << ",\n";

    // Runtime state snapshot
    std::cout << "  \"runtime_state\": {\n";
    std::cout << "    \"schema_version\": " << state_snapshot.schema_version << ",\n";
    std::cout << "    \"sequence\": " << state_snapshot.sequence << ",\n";
    std::cout << "    \"timestamp_unix_ms\": " << state_snapshot.timestamp_unix_ms << ",\n";
    std::cout << "    \"cpu_load\": " << state_snapshot.cpu_load << ",\n";
    std::cout << "    \"temperature_c\": " << state_snapshot.temperature_c << ",\n";
    std::cout << "    \"latency_budget_ms\": " << state_snapshot.latency_budget_ms << ",\n";
    std::cout << "    \"resource_pressure\": " << state_snapshot.resource_pressure << ",\n";
    std::cout << "    \"is_stale\": " << (state_snapshot.is_stale ? "true" : "false") << "\n";
    std::cout << "  },\n";

    // Decision summary
    std::cout << "  \"decision\": {\n";
    std::cout << "    \"selected_exit\": " << decision.selected_exit << ",\n";
    std::cout << "    \"selected_exit_name\": \"" << decision.selected_exit_name << "\",\n";
    std::cout << "    \"predicted_class\": " << decision.predicted_class << ",\n";
    std::cout << "    \"confidence\": " << decision.confidence << ",\n";
    std::cout << "    \"entropy\": " << decision.entropy << ",\n";
    std::cout << "    \"reason\": \"" << aeronode::ToString(decision.reason) << "\",\n";
    std::cout << "    \"evaluations\": [\n";
    for (std::size_t i = 0; i < decision.evaluations.size(); ++i) {
      const auto& ev = decision.evaluations[i];
      std::cout << "      {\n";
      std::cout << "        \"exit_index\": " << ev.exit_index << ",\n";
      std::cout << "        \"exit_name\": \"" << ev.exit_name << "\",\n";
      std::cout << "        \"entropy\": " << ev.entropy << ",\n";
      std::cout << "        \"confidence\": " << ev.confidence << ",\n";
      std::cout << "        \"threshold_used\": " << ev.threshold_used << ",\n";
      std::cout << "        \"action\": \"" << ev.action << "\",\n";
      std::cout << "        \"reason\": \"" << aeronode::ToString(ev.reason) << "\"\n";
      std::cout << "      }" << (i < decision.evaluations.size() - 1 ? "," : "") << "\n";
    }
    std::cout << "    ]\n";
    std::cout << "  },\n";

    // Per-exit raw tensors and predictions
    std::cout << "  \"exits\": [\n";
    for (std::size_t exit_idx = 0; exit_idx < 4; ++exit_idx) {
      const float* logits = outputs.at(exit_idx).GetTensorData<float>();
      std::cout << "    {\n";
      std::cout << "      \"name\": \"" << expected_names[exit_idx] << "\",\n";
      std::cout << "      \"shape\": [" << batch_size << ", 10],\n";
      std::cout << "      \"predictions\": [\n";
      for (std::int64_t b = 0; b < batch_size; ++b) {
        const float* sample_logits = logits + (b * 10);
        const auto metrics = aeronode::ComputePredictionMetrics(sample_logits, 10);
        std::cout << "        {\n";
        std::cout << "          \"sample_index\": " << b << ",\n";
        std::cout << "          \"predicted_class\": " << metrics.predicted_class << ",\n";
        std::cout << "          \"confidence\": " << metrics.confidence << ",\n";
        std::cout << "          \"entropy\": " << metrics.entropy << ",\n";
        std::cout << "          \"logits\": [";
        for (int c = 0; c < 10; ++c) {
          std::cout << sample_logits[c] << (c < 9 ? ", " : "");
        }
        std::cout << "]\n";
        std::cout << "        }" << (b < batch_size - 1 ? "," : "") << "\n";
      }
      std::cout << "      ]\n";
      std::cout << "    }" << (exit_idx < 3 ? "," : "") << "\n";
    }
    std::cout << "  ]\n";
    std::cout << "}\n";

    if (state_receiver) {
      state_receiver->Stop();
    }
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "aeronode_runtime: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

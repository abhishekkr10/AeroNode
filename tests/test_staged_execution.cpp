#include "staged_model_runner.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace {

using aeronode::StagedInferenceEngine;
const std::filesystem::path fixtures = "tmp/phase5d";
constexpr float tolerance = 1e-5F;
int passed = 0;
std::array<float, 4> maximum_errors{};

void Check(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}

template <class F>
void Test(const std::string& name, F body) {
  body();
  ++passed;
  std::cout << "[PASS] " << name << '\n';
}

template <class F>
void ExpectError(F body, const std::string& diagnostic) {
  try {
    body();
  } catch (const std::exception& error) {
    Check(std::string(error.what()).find(diagnostic) != std::string::npos,
          "Wrong diagnostic: " + std::string(error.what()) + "; expected " + diagnostic);
    return;
  }
  throw std::runtime_error("Expected rejection: " + diagnostic);
}

aeronode::StagedEngineConfig Config() {
  aeronode::StagedEngineConfig config;
  for (int i = 0; i < 4; ++i) {
    config.stage_model_paths[static_cast<std::size_t>(i)] =
        "model/artifacts/phase5/stage_" + std::to_string(i + 1) + ".onnx";
  }
  return config;
}

aeronode::RuntimeState State() {
  aeronode::RuntimeState state;
  state.latency_budget_ms = 1000000.0F;
  state.resource_pressure = 0.0F;
  return state;
}

aeronode::ExitDecisionEngine ForcedPolicy(int exit) {
  aeronode::ExitPolicyConfig config;
  // Test controls, not calibrated deployment thresholds. Entropy is >= 0 and <= ln(10).
  config.entropy_thresholds = {-1.0F, -1.0F, -1.0F, -1.0F};
  if (exit < 4) config.entropy_thresholds.at(static_cast<std::size_t>(exit - 1)) = 3.0F;
  return aeronode::ExitDecisionEngine(config);
}

std::vector<float> ReadFloats(const std::filesystem::path& path, std::size_t count) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  Check(file.is_open(), "Missing fixture " + path.string() + "; run prepare_phase5d_fixtures.py");
  Check(file.tellg() == static_cast<std::streamoff>(count * sizeof(float)), "Fixture length mismatch");
  std::vector<float> values(count);
  file.seekg(0);
  file.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
  Check(static_cast<bool>(file), "Failed reading fixture");
  return values;
}

std::vector<float> Input(int batch) {
  return ReadFloats(fixtures / ("input_" + std::to_string(batch) + ".bin"),
                    static_cast<std::size_t>(batch) * 3 * 128 * 128);
}

std::vector<std::int64_t> Shape(int batch) { return {batch, 3, 128, 128}; }

float Compare(const std::vector<float>& actual, const std::vector<float>& expected) {
  Check(actual.size() == expected.size(), "Parity element count mismatch");
  float error = 0.0F;
  for (std::size_t i = 0; i < actual.size(); ++i) {
    Check(std::isfinite(actual[i]) && std::isfinite(expected[i]), "Non-finite parity data");
    error = std::max(error, std::abs(actual[i] - expected[i]));
  }
  Check(error <= tolerance, "Absolute parity error exceeds 1e-5: " + std::to_string(error));
  for (std::size_t offset = 0; offset < actual.size(); offset += 10) {
    const auto a = std::max_element(actual.begin() + static_cast<std::ptrdiff_t>(offset),
                                    actual.begin() + static_cast<std::ptrdiff_t>(offset + 10));
    const auto e = std::max_element(expected.begin() + static_cast<std::ptrdiff_t>(offset),
                                    expected.begin() + static_cast<std::ptrdiff_t>(offset + 10));
    Check(a - actual.begin() == e - expected.begin(), "Argmax class disagreement");
  }
  return error;
}

class MonolithicReference {
 public:
  MonolithicReference() : env_(ORT_LOGGING_LEVEL_WARNING, "phase5d_reference"), session_(nullptr) {
    options_.SetIntraOpNumThreads(1);
    options_.SetInterOpNumThreads(1);
    const auto path = std::filesystem::path("model/artifacts/phase3/multi_exit_imagenette.onnx");
    session_ = Ort::Session(env_, path.c_str(), options_);
  }

  std::array<std::vector<float>, 4> Run(const std::vector<float>& input, int batch) {
    const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const auto shape = Shape(batch);
    auto tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()),
                                                  input.size(), shape.data(), shape.size());
    const char* in_names[] = {"input"};
    const char* out_names[] = {"exit_1_logits", "exit_2_logits", "exit_3_logits", "exit_4_logits"};
    auto outputs = session_.Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 4);
    std::array<std::vector<float>, 4> result;
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      Check(outputs[i].GetTensorTypeAndShapeInfo().GetShape() == std::vector<std::int64_t>{batch, 10},
            "Monolithic reference shape mismatch");
      const auto* data = outputs[i].GetTensorData<float>();
      result[i].assign(data, data + static_cast<std::size_t>(batch) * 10);
    }
    return result;
  }

 private:
  Ort::Env env_;
  Ort::SessionOptions options_;
  Ort::Session session_;
};

void CheckTrace(const aeronode::StagedInferenceResult& result, int exit, int batch) {
  Check(result.selected_exit == exit, "Wrong exit selected");
  Check(result.output_shape == std::vector<std::int64_t>{batch, 10}, "Returned shape mismatch");
  Check(result.logits.size() == static_cast<std::size_t>(batch) * 10, "Returned count mismatch");
  Check(result.stages.size() == static_cast<std::size_t>(exit), "Wrong number of executed stages");
  const std::array<std::vector<std::int64_t>, 4> inputs = {
      Shape(batch), std::vector<std::int64_t>{batch, 32, 32, 32},
      std::vector<std::int64_t>{batch, 64, 16, 16}, std::vector<std::int64_t>{batch, 128, 8, 8}};
  for (int i = 0; i < 4; ++i) {
    Check(result.stage_run_counts[static_cast<std::size_t>(i)] == (i < exit ? 1U : 0U),
          "Actual ORT Run counter mismatch at stage " + std::to_string(i + 1));
    if (i >= exit) continue;
    const auto& stage = result.stages[static_cast<std::size_t>(i)];
    Check(stage.stage_index == i + 1 && stage.input_shape == inputs[static_cast<std::size_t>(i)],
          "Stage input dimensions mismatch");
    Check(stage.output_shapes.size() == (i < 3 ? 2U : 1U), "Output count mismatch");
    Check(stage.output_shapes.back() == std::vector<std::int64_t>{batch, 10}, "Exit shape mismatch");
    if (i < 3) Check(stage.output_shapes[0] == inputs[static_cast<std::size_t>(i + 1)], "Feature shape mismatch");
    const auto expected_copy = i + 1 < exit ?
        aeronode::IntermediateFallbackBuffers::CalculateElementCount(i + 1, batch) : 0;
    Check(stage.copied_feature_elements == expected_copy, "Wrong copied element count");
    Check(stage.sample_evaluations.size() == static_cast<std::size_t>(batch), "Missing sample decisions");
    for (const auto& eval : stage.sample_evaluations) {
      Check(eval.exit_index == i + 1 && eval.exited == (i + 1 == exit), "Wrong stage policy evaluation");
    }
  }
}

}  // namespace

int main() {
  try {
    StagedInferenceEngine engine(Config());
    MonolithicReference mono;
    engine.WarmUp();
    for (const int batch : {1, 3, 5}) {
      const auto input = Input(batch);
      const auto reference = mono.Run(input, batch);
      for (int exit = 1; exit <= 4; ++exit) {
        Test("forced_exit_" + std::to_string(exit) + "_batch_" + std::to_string(batch), [&] {
          const auto result = engine.Infer(input, Shape(batch), State(), ForcedPolicy(exit));
          CheckTrace(result, exit, batch);
          const auto python_reference = ReadFloats(fixtures / ("reference_" + std::to_string(batch) +
              "_" + std::to_string(exit) + ".bin"), static_cast<std::size_t>(batch) * 10);
          const auto i = static_cast<std::size_t>(exit - 1);
          maximum_errors[i] = std::max(maximum_errors[i], Compare(result.logits, reference[i]));
          maximum_errors[i] = std::max(maximum_errors[i], Compare(result.logits, python_reference));
        });
      }
    }
    const auto input = Input(1);
    Test("owned_results_and_batch_resize_reuse", [&] {
      const auto original = engine.Infer(input, Shape(1), State(), ForcedPolicy(4));
      const auto retained = original.logits;
      for (const int batch : {5, 1, 3, 1}) {
        engine.Infer(Input(batch), Shape(batch), State(), ForcedPolicy(4));
      }
      Check(original.logits == retained, "Previous result changed after scratch reuse");
      Compare(original.logits, mono.Run(input, 1)[3]);
    });
    Test("serialized_concurrent_requests", [&] {
      auto request = [&](int batch) { return engine.Infer(Input(batch), Shape(batch), State(), ForcedPolicy(4)); };
      auto a = std::async(std::launch::async, request, 1);
      auto b = std::async(std::launch::async, request, 5);
      Compare(a.get().logits, mono.Run(Input(1), 1)[3]);
      Compare(b.get().logits, mono.Run(Input(5), 5)[3]);
    });
    Test("explicit_buffers_match_independent_stage_outputs", [&] {
      engine.Infer(input, Shape(1), State(), ForcedPolicy(4));
      const auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      auto shape = Shape(1);
      auto tensor = Ort::Value::CreateTensor<float>(memory, const_cast<float*>(input.data()), input.size(),
                                                    shape.data(), shape.size());
      const std::array<const std::vector<float>*, 3> buffers = {&engine.GetFallbackBuffers().feat_1,
          &engine.GetFallbackBuffers().feat_2, &engine.GetFallbackBuffers().feat_3};
      for (std::size_t i = 0; i < 3; ++i) {
        const auto& meta = engine.GetStageMetadata(i);
        const char* in_names[] = {meta.inputs[0].name.c_str()};
        const char* out_names[] = {meta.outputs[0].name.c_str(), meta.outputs[1].name.c_str()};
        auto outputs = engine.GetSession(i).Run(Ort::RunOptions{nullptr}, in_names, &tensor, 1, out_names, 2);
        const auto* feature = outputs[0].GetTensorData<float>();
        Check(buffers[i]->data() != feature, "Buffer aliases ORT output");
        Check(buffers[i]->size() == outputs[0].GetTensorTypeAndShapeInfo().GetElementCount(), "Buffer size mismatch");
        Check(std::equal(buffers[i]->begin(), buffers[i]->end(), feature), "Explicit copy contents mismatch");
        tensor = engine.CreateFallbackInputTensor(static_cast<int>(i + 2), 1);
      }
      Check(buffers[0]->data() != buffers[1]->data() && buffers[1]->data() != buffers[2]->data(),
            "Scratch buffers alias each other");
    });
    Test("latency_policy_stops_real_execution", [&] {
      auto state = State();
      state.latency_budget_ms = 0.001F;
      const auto result = engine.Infer(input, Shape(1), state, ForcedPolicy(4));
      CheckTrace(result, 1, 1);
      Check(result.stages[0].sample_evaluations[0].reason == aeronode::DecisionReason::LATENCY_BUDGET_EXHAUSTED,
            "Wrong budget reason");
    });
    Test("pressure_policy_stops_real_execution", [&] {
      const auto reference = mono.Run(input, 1);
      const auto metrics = aeronode::ComputePredictionMetrics(reference[0]);
      aeronode::ExitPolicyConfig policy;
      policy.entropy_thresholds[0] = metrics.entropy - 0.15F;
      auto state = State();
      state.resource_pressure = 0.9F;
      const auto result = engine.Infer(input, Shape(1), state, aeronode::ExitDecisionEngine(policy));
      CheckTrace(result, 1, 1);
      Check(result.stages[0].sample_evaluations[0].reason == aeronode::DecisionReason::RESOURCE_PRESSURE_FORCED_EXIT,
            "Wrong pressure reason");
    });
    Test("mixed_batch_waits_for_every_sample", [&] {
      const auto data = Input(5);
      const auto reference = mono.Run(data, 5);
      std::array<float, 5> entropy{};
      for (std::size_t i = 0; i < entropy.size(); ++i) {
        entropy[i] = aeronode::ComputePredictionMetrics(reference[0].data() + i * 10, 10).entropy;
      }
      const auto range = std::minmax_element(entropy.begin(), entropy.end());
      Check(*range.second > *range.first, "Mixed-batch fixture has identical entropies");
      aeronode::ExitPolicyConfig policy;
      policy.entropy_thresholds = {(*range.first + *range.second) / 2.0F, 3.0F, -1.0F, -1.0F};
      const auto result = engine.Infer(data, Shape(5), State(), aeronode::ExitDecisionEngine(policy));
      Check(result.selected_exit == 2 && result.stage_run_counts == std::array<std::size_t, 4>{1, 1, 0, 0},
            "Mixed batch exited prematurely");
      const auto& evaluations = result.stages[0].sample_evaluations;
      Check(std::any_of(evaluations.begin(), evaluations.end(), [](const auto& e) { return e.exited; }) &&
            std::any_of(evaluations.begin(), evaluations.end(), [](const auto& e) { return !e.exited; }),
            "Missing mixed decisions");
      Compare(result.logits, reference[1]);
    });
    Test("incremental_policy_matches_phase4", [&] {
      const aeronode::ExitDecisionEngine policy;
      for (const float pressure : {0.0F, 0.8F, 0.9F}) {
        for (const float budget : {1.0F, 3.0F, 1000.0F}) {
          for (const float entropy : {0.0F, 0.6F, 0.8F, 1.1F, 2.3F}) {
            auto state = State(); state.resource_pressure = pressure; state.latency_budget_ms = budget;
            std::vector<aeronode::PredictionMetrics> metrics(4);
            for (auto& m : metrics) m.entropy = entropy;
            const auto full = policy.Evaluate(metrics, state, 1.0);
            for (const auto& expected : full.evaluations) {
              const auto step = policy.EvaluateStage(metrics[static_cast<std::size_t>(expected.exit_index - 1)],
                                                     expected.exit_index, state, 1.0);
              Check(step.exited == expected.exited && step.reason == expected.reason &&
                    step.action == expected.action && step.threshold_used == expected.threshold_used,
                    "Incremental policy changed Phase-4 behavior");
            }
          }
        }
      }
      aeronode::PredictionMetrics uncertain; uncertain.entropy = 2.3F;
      Check(!policy.EvaluateStage(uncertain, 1, State()).exited, "Available head treated as final");
      ExpectError([&] { policy.EvaluateStage(uncertain, 0, State()); }, "exit_index");
      ExpectError([&] { policy.EvaluateStage(uncertain, 5, State()); }, "exit_index");
    });

    const std::array<std::vector<std::int64_t>, 6> bad_shapes = {
        std::vector<std::int64_t>{}, {1, 3, 128}, {0, 3, 128, 128}, {-1, 3, 128, 128},
        {1, 3, 127, 128}, {1, 4, 128, 128}};
    for (std::size_t i = 0; i < bad_shapes.size(); ++i) {
      Test("invalid_input_shape_" + std::to_string(i), [&] {
        ExpectError([&] { engine.Infer(input, bad_shapes[i], State()); }, "expected [batch");
      });
    }
    Test("empty_input", [&] { ExpectError([&] { engine.Infer({}, Shape(1), State()); }, "element count"); });
    Test("incorrect_input_element_count", [&] {
      auto short_input = input; short_input.pop_back();
      ExpectError([&] { engine.Infer(short_input, Shape(1), State()); }, "element count");
    });
    Test("overflow_protection", [&] {
      ExpectError([&] { engine.Infer(input, {std::numeric_limits<std::int64_t>::max(), 3, 128, 128}, State()); }, "overflow");
      ExpectError([&] { aeronode::IntermediateFallbackBuffers::CalculateByteSize(1, std::numeric_limits<std::int64_t>::max()); }, "overflow");
    });
    for (const auto value : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
      Test(std::isnan(value) ? "nan_input" : "infinite_input", [&] {
        auto bad = input; bad[0] = value;
        ExpectError([&] { engine.Infer(bad, Shape(1), State()); }, "non-finite");
      });
    }
    Test("invalid_state", [&] {
      auto state = State(); state.latency_budget_ms = 0;
      ExpectError([&] { engine.Infer(input, Shape(1), state); }, "RuntimeState");
    });
    Test("scratch_input_alias", [&] {
      auto& buffer = engine.GetFallbackBuffers().feat_1; buffer = input;
      ExpectError([&] { engine.Infer(buffer, Shape(1), State()); }, "alias");
    });
    Test("successful_request_after_input_errors", [&] {
      Compare(engine.Infer(input, Shape(1), State(), ForcedPolicy(4)).logits, mono.Run(input, 1)[3]);
    });
    for (const auto& path : {std::string(""), std::string("missing_stage.onnx"), (fixtures / "corrupt.onnx").string()}) {
      Test("invalid_model_path_" + std::to_string(passed), [&] {
        auto config = Config(); config.stage_model_paths[1] = path;
        ExpectError([&] { StagedInferenceEngine invalid(config); }, path.empty() ? "path is empty" : "stage 2");
      });
    }
    for (const auto& item : std::array<std::pair<const char*, const char*>, 12>{{
        {"bad_feature_metadata.onnx", "metadata shape mismatch"}, {"bad_logits_metadata.onnx", "metadata shape mismatch"},
        {"bad_rank_metadata.onnx", "metadata shape mismatch"}, {"bad_type_metadata.onnx", "not float32"},
        {"bad_batch_metadata.onnx", "dynamic input batch"}, {"bad_input_metadata.onnx", "Input shape mismatch"},
        {"bad_output_count.onnx", "Output count mismatch"},
        {"bad_stage_identity.onnx", "malformed model metadata 'stage_index'"},
        {"bad_dynamic_property.onnx", "malformed model metadata 'dynamic_batch'"},
        {"bad_hash_metadata.onnx", "malformed source_checkpoint_sha256"},
        {"mixed_checkpoint.onnx", "source checkpoint hash mismatch"},
        {"missing_metadata.onnx", "missing model metadata"}}}) {
      Test(item.first, [&] {
        auto config = Config(); config.stage_model_paths[1] = (fixtures / item.first).string();
        ExpectError([&] { StagedInferenceEngine invalid(config); }, item.second);
      });
    }
    Test("wrong_stage_order", [&] {
      auto config = Config(); std::swap(config.stage_model_paths[1], config.stage_model_paths[2]);
      ExpectError([&] { StagedInferenceEngine invalid(config); }, "Input name mismatch");
    });
    for (const auto& item : std::array<std::pair<const char*, const char*>, 5>{{
        {"bad_feature_shape.onnx", "feat_2': shape mismatch"}, {"bad_logits_shape.onnx", "exit_2_logits': shape mismatch"},
        {"bad_feature_finite.onnx", "non-finite"}, {"bad_logits_finite.onnx", "non-finite"},
        {"run_failure.onnx", "stage_2 ONNX Runtime Run failed"}}}) {
      Test(item.first, [&] {
        auto config = Config(); config.stage_model_paths[1] = (fixtures / item.first).string();
        StagedInferenceEngine invalid(config);
        ExpectError([&] { invalid.Infer(input, Shape(1), State(), ForcedPolicy(4)); }, item.second);
        // The identical failing downstream model remains loaded, yet Exit 1 succeeds.
        const auto early = invalid.Infer(input, Shape(1), State(), ForcedPolicy(1));
        CheckTrace(early, 1, 1);
        Compare(early.logits, mono.Run(input, 1)[0]);
      });
    }
    for (std::size_t i = 0; i < maximum_errors.size(); ++i) {
      std::cout << "Exit " << i + 1 << " max absolute error: " << maximum_errors[i] << '\n';
    }
    std::cout << "All " << passed << " Phase 5D C++ tests passed.\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "Phase 5D failure after " << passed << " passed tests: " << error.what() << '\n';
    return 1;
  }
}

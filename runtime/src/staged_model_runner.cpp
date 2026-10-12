#include "staged_model_runner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace aeronode {

namespace {

std::string ShapeString(const std::vector<std::int64_t>& shape) {
  std::ostringstream out;
  out << '[';
  for (std::size_t i = 0; i < shape.size(); ++i) {
    if (i != 0) out << ',';
    out << shape[i];
  }
  out << ']';
  return out.str();
}

std::size_t CheckedElementCount(const std::vector<std::int64_t>& shape) {
  if (shape.empty()) throw std::invalid_argument("Tensor shape is empty");
  std::size_t count = 1;
  for (const auto dim : shape) {
    if (dim <= 0) throw std::invalid_argument("Tensor dimensions must be positive: " + ShapeString(shape));
    if (static_cast<std::uint64_t>(dim) > std::numeric_limits<std::size_t>::max()) {
      throw std::overflow_error("Tensor dimension exceeds addressable size: " + ShapeString(shape));
    }
    const auto size = static_cast<std::size_t>(dim);
    if (count > std::numeric_limits<std::size_t>::max() / size) {
      throw std::overflow_error("Tensor element count overflow: " + ShapeString(shape));
    }
    count *= size;
  }
  if (count > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
    throw std::overflow_error("Tensor byte count overflow: " + ShapeString(shape));
  }
  return count;
}

std::vector<std::int64_t> BatchShape(const TensorMetadata& meta, std::int64_t batch) {
  auto shape = meta.shape;
  shape.at(0) = batch;
  return shape;
}

void ValidateFinite(const float* data, std::size_t count, const std::string& context) {
  if (data == nullptr || count == 0) throw std::invalid_argument(context + ": empty tensor buffer");
  for (std::size_t i = 0; i < count; ++i) {
    if (!std::isfinite(data[i])) {
      throw std::invalid_argument(context + ": non-finite value at element " + std::to_string(i));
    }
  }
}

void ValidateTensor(const Ort::Value& tensor, const std::vector<std::int64_t>& expected,
                    const std::string& context) {
  if (!tensor.IsTensor()) throw std::runtime_error(context + ": expected a tensor");
  const auto info = tensor.GetTensorTypeAndShapeInfo();
  if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    throw std::runtime_error(context + ": expected float32");
  }
  const auto actual = info.GetShape();
  if (actual != expected) {
    throw std::runtime_error(context + ": shape mismatch; expected " + ShapeString(expected) +
                             ", got " + ShapeString(actual));
  }
  const auto count = CheckedElementCount(expected);
  if (info.GetElementCount() != count) {
    throw std::runtime_error(context + ": element count mismatch; expected " + std::to_string(count) +
                             ", got " + std::to_string(info.GetElementCount()));
  }
  ValidateFinite(tensor.GetTensorData<float>(), count, context);
}

}  // namespace

// ---------------------------------------------------------------------------
// IntermediateFallbackBuffers Implementation
// ---------------------------------------------------------------------------

std::size_t IntermediateFallbackBuffers::CalculateElementCount(int stage_index, std::int64_t batch_size) {
  if (batch_size <= 0) {
    throw std::invalid_argument("batch_size must be positive, got " + std::to_string(batch_size));
  }
  switch (stage_index) {
    case 1:
      // feat_1: [B, 32, 32, 32]
      return CheckedElementCount({batch_size, 32, 32, 32});
    case 2:
      // feat_2: [B, 64, 16, 16]
      return CheckedElementCount({batch_size, 64, 16, 16});
    case 3:
      // feat_3: [B, 128, 8, 8]
      return CheckedElementCount({batch_size, 128, 8, 8});
    default:
      throw std::invalid_argument(
          "Invalid stage index for intermediate fallback buffer: " + std::to_string(stage_index) +
          " (valid intermediate stages are 1, 2, 3)");
  }
}

std::size_t IntermediateFallbackBuffers::CalculateByteSize(int stage_index, std::int64_t batch_size) {
  return CalculateElementCount(stage_index, batch_size) * sizeof(float);
}

void IntermediateFallbackBuffers::ResizeForBatch(std::int64_t batch_size) {
  feat_1.resize(CalculateElementCount(1, batch_size));
  feat_2.resize(CalculateElementCount(2, batch_size));
  feat_3.resize(CalculateElementCount(3, batch_size));
}

// ---------------------------------------------------------------------------
// StagedInferenceEngine Implementation
// ---------------------------------------------------------------------------

StagedInferenceEngine::StagedInferenceEngine(const StagedEngineConfig& config)
    : config_(config),
      environment_(ORT_LOGGING_LEVEL_WARNING, "aeronode_staged"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
  session_options_.SetGraphOptimizationLevel(config_.optimization_level);
  session_options_.SetIntraOpNumThreads(config_.intra_op_threads);
  session_options_.SetInterOpNumThreads(config_.inter_op_threads);

  InitializeSessions();
  InspectAndValidateContracts();
  is_initialized_ = true;
}

void StagedInferenceEngine::InitializeSessions() {
  for (std::size_t i = 0; i < 4; ++i) {
    const std::string& model_path = config_.stage_model_paths[i];
    if (model_path.empty()) {
      throw std::invalid_argument("Stage " + std::to_string(i + 1) + " model path is empty");
    }

    try {
#ifdef _WIN32
      const auto native_path = std::filesystem::u8path(model_path);
      sessions_[i] = std::make_unique<Ort::Session>(environment_, native_path.c_str(), session_options_);
#else
      sessions_[i] = std::make_unique<Ort::Session>(environment_, model_path.c_str(), session_options_);
#endif
    } catch (const Ort::Exception& err) {
      throw std::runtime_error(
          "Failed to load ONNX Runtime session for stage " + std::to_string(i + 1) + " ('" +
          model_path + "'): " + err.what());
    }
  }
}

void StagedInferenceEngine::InspectAndValidateContracts() {
  Ort::AllocatorWithDefaultOptions allocator;
  std::string checkpoint_hash;

  // Expected contracts for Stage 1 through Stage 4
  struct ExpectedStageContract {
    std::string input_name;
    std::vector<std::int64_t> input_suffix;  // e.g. {3, 128, 128}
    std::vector<std::string> output_names;
  };

  const std::array<ExpectedStageContract, 4> expected_contracts = {{
      {"input", {3, 128, 128}, {"feat_1", "exit_1_logits"}},
      {"feat_1", {32, 32, 32}, {"feat_2", "exit_2_logits"}},
      {"feat_2", {64, 16, 16}, {"feat_3", "exit_3_logits"}},
      {"feat_3", {128, 8, 8}, {"exit_4_logits"}},
  }};

  for (std::size_t stage_idx = 0; stage_idx < 4; ++stage_idx) {
    auto& session = sessions_[stage_idx];
    StageMetadata& meta = metadata_[stage_idx];
    meta.stage_index = static_cast<int>(stage_idx + 1);
    meta.stage_name = "stage_" + std::to_string(meta.stage_index);

    const std::size_t input_count = session->GetInputCount();
    const std::size_t output_count = session->GetOutputCount();

    // Inspect Inputs
    meta.inputs.reserve(input_count);
    for (std::size_t i = 0; i < input_count; ++i) {
      TensorMetadata tm;
      const auto name_ptr = session->GetInputNameAllocated(i, allocator);
      tm.name = name_ptr.get();
      // GetTensorTypeAndShapeInfo is an unowned view of TypeInfo. Keep its owner alive.
      const auto owner = session->GetInputTypeInfo(i);
      if (owner.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error(meta.stage_name + ": input metadata is not a tensor");
      }
      const auto info = owner.GetTensorTypeAndShapeInfo();
      tm.shape = info.GetShape();
      tm.element_type = info.GetElementType();
      meta.inputs.push_back(std::move(tm));
    }

    // Inspect Outputs
    meta.outputs.reserve(output_count);
    for (std::size_t i = 0; i < output_count; ++i) {
      TensorMetadata tm;
      const auto name_ptr = session->GetOutputNameAllocated(i, allocator);
      tm.name = name_ptr.get();
      const auto owner = session->GetOutputTypeInfo(i);
      if (owner.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error(meta.stage_name + ": output metadata is not a tensor");
      }
      const auto info = owner.GetTensorTypeAndShapeInfo();
      tm.shape = info.GetShape();
      tm.element_type = info.GetElementType();
      meta.outputs.push_back(std::move(tm));
    }

    // Validate against expected contract
    const auto& expected = expected_contracts[stage_idx];

    // Check input count
    if (meta.inputs.size() != 1) {
      throw std::runtime_error(
          "Contract violation in " + meta.stage_name + ": expected exactly 1 input, found " +
          std::to_string(meta.inputs.size()));
    }

    // Check input name
    if (meta.inputs[0].name != expected.input_name) {
      throw std::runtime_error(
          "Input name mismatch in " + meta.stage_name + ": expected '" + expected.input_name +
          "', got '" + meta.inputs[0].name + "'");
    }

    // Check input element type
    if (meta.inputs[0].element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      throw std::runtime_error("Input tensor in " + meta.stage_name + " is not float32");
    }

    // Check input shape rank and spatial dimensions
    const auto& in_shape = meta.inputs[0].shape;
    if (in_shape.size() != 4) {
      throw std::runtime_error(
          "Input tensor in " + meta.stage_name + " must have rank 4, got rank " +
          std::to_string(in_shape.size()));
    }
    if (in_shape[0] != -1) {
      throw std::runtime_error(meta.stage_name + ": expected dynamic input batch dimension");
    }
    for (std::size_t d = 0; d < 3; ++d) {
      if (in_shape[d + 1] != expected.input_suffix[d]) {
        throw std::runtime_error(
            "Input shape mismatch in " + meta.stage_name + " at dimension " + std::to_string(d + 1) +
            ": expected " + std::to_string(expected.input_suffix[d]) + ", got " +
            std::to_string(in_shape[d + 1]));
      }
    }

    // Check output count
    if (meta.outputs.size() != expected.output_names.size()) {
      throw std::runtime_error(
          "Output count mismatch in " + meta.stage_name + ": expected " +
          std::to_string(expected.output_names.size()) + ", found " +
          std::to_string(meta.outputs.size()));
    }

    // Check output names and types
    for (std::size_t out_idx = 0; out_idx < expected.output_names.size(); ++out_idx) {
      if (meta.outputs[out_idx].name != expected.output_names[out_idx]) {
        throw std::runtime_error(
            "Output name mismatch in " + meta.stage_name + " at index " + std::to_string(out_idx) +
            ": expected '" + expected.output_names[out_idx] + "', got '" +
            meta.outputs[out_idx].name + "'");
      }
      if (meta.outputs[out_idx].element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        throw std::runtime_error(
            "Output tensor '" + meta.outputs[out_idx].name + "' in " + meta.stage_name +
            " is not float32");
      }
      const std::vector<std::int64_t> expected_shape =
          (stage_idx < 3 && out_idx == 0)
              ? std::vector<std::int64_t>{-1, expected_contracts[stage_idx + 1].input_suffix[0],
                                         expected_contracts[stage_idx + 1].input_suffix[1],
                                         expected_contracts[stage_idx + 1].input_suffix[2]}
              : std::vector<std::int64_t>{-1, 10};
      if (meta.outputs[out_idx].shape != expected_shape) {
        throw std::runtime_error(meta.stage_name + " output '" + meta.outputs[out_idx].name +
                                 "': metadata shape mismatch; expected " + ShapeString(expected_shape) +
                                 ", got " + ShapeString(meta.outputs[out_idx].shape));
      }
    }

    // Phase-5A exports carry stage identity and checkpoint provenance. Reject
    // mislabeled/mixed artifacts even when their tensor dimensions happen to fit.
    const auto model_meta = session->GetModelMetadata();
    const auto property = [&](const char* key) {
      const auto value = model_meta.LookupCustomMetadataMapAllocated(key, allocator);
      if (!value) throw std::runtime_error(meta.stage_name + ": missing model metadata '" + key + "'");
      return std::string(value.get());
    };
    for (const auto& entry : std::array<std::pair<const char*, std::string>, 4>{{
             {"aeronode_phase", "5A"}, {"stage_index", std::to_string(stage_idx + 1)},
             {"stage_name", meta.stage_name}, {"dynamic_batch", "true"}}}) {
      if (property(entry.first) != entry.second) {
        throw std::runtime_error(meta.stage_name + ": malformed model metadata '" + entry.first + "'");
      }
    }
    const auto hash = property("source_checkpoint_sha256");
    if (hash.size() != 64 || hash.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
      throw std::runtime_error(meta.stage_name + ": malformed source_checkpoint_sha256");
    }
    if (stage_idx == 0) checkpoint_hash = hash;
    if (hash != checkpoint_hash) {
      throw std::runtime_error(meta.stage_name + ": source checkpoint hash mismatch across stages");
    }
  }
}

void StagedInferenceEngine::WarmUp() {
  const std::lock_guard<std::mutex> lock(inference_mutex_);
  is_warmed_up_ = false;
  if (!is_initialized_) {
    throw std::runtime_error("Cannot warm up uninitialized StagedInferenceEngine");
  }

  // Define dummy input tensor shapes for batch_size=1
  const std::array<std::vector<std::int64_t>, 4> dummy_shapes = {{
      {1, 3, 128, 128},   // Stage 1
      {1, 32, 32, 32},    // Stage 2
      {1, 64, 16, 16},    // Stage 3
      {1, 128, 8, 8},     // Stage 4
  }};

  for (std::size_t stage_idx = 0; stage_idx < 4; ++stage_idx) {
    const auto& shape = dummy_shapes[stage_idx];
    const std::size_t element_count = static_cast<std::size_t>(shape[0] * shape[1] * shape[2] * shape[3]);
    std::vector<float> dummy_data(element_count, 0.0F);

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_, dummy_data.data(), dummy_data.size(), shape.data(), shape.size());

    RunStage(stage_idx, input_tensor);
  }

  is_warmed_up_ = true;
}

std::vector<Ort::Value> StagedInferenceEngine::RunStage(
    std::size_t stage_index, const Ort::Value& input, std::size_t* run_count) {
  const auto& meta = metadata_.at(stage_index);
  const auto batch = input.GetTensorTypeAndShapeInfo().GetShape().at(0);
  ValidateTensor(input, BatchShape(meta.inputs[0], batch), meta.stage_name + " input");
  const char* in_names[] = {meta.inputs[0].name.c_str()};
  std::array<const char*, 2> out_names{};
  for (std::size_t i = 0; i < meta.outputs.size(); ++i) out_names[i] = meta.outputs[i].name.c_str();
  try {
    // Instrument the actual ORT call, rather than deriving counts from the exit number.
    if (run_count != nullptr) ++*run_count;
    auto outputs = sessions_[stage_index]->Run(Ort::RunOptions{nullptr}, in_names, &input, 1,
                                               out_names.data(), meta.outputs.size());
    if (outputs.size() != meta.outputs.size()) {
      throw std::runtime_error(meta.stage_name + ": unexpected runtime output count; expected " +
                               std::to_string(meta.outputs.size()) + ", got " + std::to_string(outputs.size()));
    }
    for (std::size_t i = 0; i < outputs.size(); ++i) {
      ValidateTensor(outputs[i], BatchShape(meta.outputs[i], batch),
                     meta.stage_name + " output '" + meta.outputs[i].name + "'");
    }
    return outputs;
  } catch (const Ort::Exception& err) {
    throw std::runtime_error(meta.stage_name + " ONNX Runtime Run failed: " + err.what());
  }
}

StagedInferenceResult StagedInferenceEngine::Infer(
    const std::vector<float>& input, const std::vector<std::int64_t>& shape,
    RuntimeState state, const ExitDecisionEngine& policy) {
  const std::lock_guard<std::mutex> lock(inference_mutex_);
  if (shape.size() != 4 || shape[0] <= 0 || shape[1] != 3 || shape[2] != 128 || shape[3] != 128) {
    throw std::invalid_argument("stage_1 input: expected [batch>0,3,128,128], got " + ShapeString(shape));
  }
  const auto expected_count = CheckedElementCount(shape);
  if (input.empty() || input.size() != expected_count) {
    throw std::invalid_argument("stage_1 input: element count mismatch; expected " +
                               std::to_string(expected_count) + ", got " + std::to_string(input.size()));
  }
  if (&input == &fallback_buffers_.feat_1 || &input == &fallback_buffers_.feat_2 ||
      &input == &fallback_buffers_.feat_3) {
    throw std::invalid_argument("stage_1 input must not alias engine scratch buffers");
  }
  if (!state.IsValid()) throw std::invalid_argument("Invalid RuntimeState snapshot");
  ValidateFinite(input.data(), input.size(), "stage_1 input");

  StagedInferenceResult result;
  result.stages.reserve(4);
  // ORT borrows this caller-owned buffer synchronously and treats Run inputs as read-only.
  auto tensor = Ort::Value::CreateTensor<float>(memory_info_, const_cast<float*>(input.data()),
                                               input.size(), shape.data(), shape.size());
  const auto start = std::chrono::steady_clock::now();
  for (std::size_t stage = 0; stage < 4; ++stage) {
    auto outputs = RunStage(stage, tensor, &result.stage_run_counts[stage]);
    StageExecution trace;
    trace.stage_index = static_cast<int>(stage + 1);
    trace.input_shape = tensor.GetTensorTypeAndShapeInfo().GetShape();
    trace.output_shapes.reserve(outputs.size());
    trace.sample_evaluations.reserve(static_cast<std::size_t>(shape[0]));
    for (const auto& output : outputs) trace.output_shapes.push_back(output.GetTensorTypeAndShapeInfo().GetShape());
    const auto& logits_tensor = outputs.back();
    const float* logits = logits_tensor.GetTensorData<float>();
    const double elapsed_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    bool all_exit = true;
    for (std::int64_t sample = 0; sample < shape[0]; ++sample) {
      const auto metrics = ComputePredictionMetrics(logits + static_cast<std::size_t>(sample) * 10, 10);
      if (!std::isfinite(metrics.entropy) || !std::isfinite(metrics.confidence)) {
        throw std::runtime_error("Non-finite prediction metrics in stage_" + std::to_string(stage + 1));
      }
      trace.sample_evaluations.push_back(policy.EvaluateStage(metrics, trace.stage_index, state, elapsed_ms));
      all_exit = all_exit && trace.sample_evaluations.back().exited;
    }
    if (all_exit) {
      result.selected_exit = trace.stage_index;
      result.output_shape = trace.output_shapes.back();
      const auto count = logits_tensor.GetTensorTypeAndShapeInfo().GetElementCount();
      result.logits.assign(logits, logits + count);
      result.stages.push_back(std::move(trace));
      return result;  // No downstream Session::Run is reached.
    }

    // Copy into a distinct engine-owned vector while this stage's outputs are
    // alive. Next session receives a fresh Ort::Value wrapping only that vector.
    auto& buffer = stage == 0 ? fallback_buffers_.feat_1 :
                   stage == 1 ? fallback_buffers_.feat_2 : fallback_buffers_.feat_3;
    const auto count = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    const auto expected_feature_count = IntermediateFallbackBuffers::CalculateElementCount(
        static_cast<int>(stage + 1), shape[0]);
    if (count == 0 || count != expected_feature_count) {
      throw std::runtime_error("Intermediate feature element count mismatch in stage_" + std::to_string(stage + 1));
    }
    buffer.resize(count);
    std::copy_n(outputs[0].GetTensorData<float>(), count, buffer.data());
    trace.copied_feature_elements = count;
    tensor = CreateFallbackInputTensor(static_cast<int>(stage + 2), shape[0]);
    result.stages.push_back(std::move(trace));
  }
  throw std::logic_error("Staged inference did not reach a terminal exit");
}

const StageMetadata& StagedInferenceEngine::GetStageMetadata(std::size_t stage_index) const {
  if (stage_index >= 4) {
    throw std::out_of_range("Stage index out of range: " + std::to_string(stage_index));
  }
  return metadata_[stage_index];
}

Ort::Session& StagedInferenceEngine::GetSession(std::size_t stage_index) {
  if (stage_index >= 4 || !sessions_[stage_index]) {
    throw std::out_of_range("Invalid session access at stage index " + std::to_string(stage_index));
  }
  return *sessions_[stage_index];
}

const Ort::Session& StagedInferenceEngine::GetSession(std::size_t stage_index) const {
  if (stage_index >= 4 || !sessions_[stage_index]) {
    throw std::out_of_range("Invalid session access at stage index " + std::to_string(stage_index));
  }
  return *sessions_[stage_index];
}

Ort::Value StagedInferenceEngine::CreateFallbackInputTensor(int stage_index, std::int64_t batch_size) {
  std::vector<std::int64_t> shape;
  float* data_ptr = nullptr;
  std::size_t element_count = 0;

  switch (stage_index) {
    case 2:
      shape = {batch_size, 32, 32, 32};
      element_count = IntermediateFallbackBuffers::CalculateElementCount(1, batch_size);
      fallback_buffers_.feat_1.resize(element_count);
      data_ptr = fallback_buffers_.feat_1.data();
      break;
    case 3:
      shape = {batch_size, 64, 16, 16};
      element_count = IntermediateFallbackBuffers::CalculateElementCount(2, batch_size);
      fallback_buffers_.feat_2.resize(element_count);
      data_ptr = fallback_buffers_.feat_2.data();
      break;
    case 4:
      shape = {batch_size, 128, 8, 8};
      element_count = IntermediateFallbackBuffers::CalculateElementCount(3, batch_size);
      fallback_buffers_.feat_3.resize(element_count);
      data_ptr = fallback_buffers_.feat_3.data();
      break;
    default:
      throw std::invalid_argument(
          "CreateFallbackInputTensor only supports intermediate stages 2, 3, 4 (got " +
          std::to_string(stage_index) + ")");
  }

  return Ort::Value::CreateTensor<float>(
      memory_info_, data_ptr, element_count, shape.data(), shape.size());
}

}  // namespace aeronode

#pragma once

#include <onnxruntime_cxx_api.h>

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "exit_decision.hpp"

namespace aeronode {

/**
 * Tensor metadata descriptor for safe introspection.
 */
struct TensorMetadata {
  std::string name;
  std::vector<std::int64_t> shape;
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
};

/**
 * Stage metadata descriptor containing input and output tensor contracts.
 */
struct StageMetadata {
  int stage_index = 0;  // 1 to 4
  std::string stage_name;
  std::vector<TensorMetadata> inputs;
  std::vector<TensorMetadata> outputs;
};

/**
 * Configuration options for the StagedInferenceEngine.
 */
struct StagedEngineConfig {
  std::array<std::string, 4> stage_model_paths;
  int intra_op_threads = 1;
  int inter_op_threads = 1;
  GraphOptimizationLevel optimization_level = GraphOptimizationLevel::ORT_ENABLE_ALL;
};

/**
 * Explicit-copy fallback scratch buffers for intermediate activations.
 *
 * MEMORY, ALLOCATOR, AND LIFETIME ARCHITECTURE:
 * 1. Allocator Independence: Each Ort::Session allocates memory from its own internal
 *    arena. Sharing an Ort::Env manages background worker threads and logging, but
 *    does NOT guarantee unified memory arenas or safe pointer transfer across sessions.
 * 2. Lifetime Guarantees: Buffers inside IntermediateFallbackBuffers are managed as
 *    contiguous std::vector<float> instances owned by StagedInferenceEngine via RAII.
 * 3. Memory Isolation: When copying an intermediate activation across stage boundaries,
 *    payload bytes are copied into these explicit fallback buffers, ensuring that
 *    Stage k's buffer deallocation or arena reuse cannot invalidate Stage k+1's input.
 * 4. Zero-Copy Disclaimer: No zero-copy claims are made. Correctness and memory safety
 *    are prioritized over premature optimization.
 */
struct IntermediateFallbackBuffers {
  std::vector<float> feat_1;  // Intermediate activation for Stage 1 -> Stage 2: [B, 32, 32, 32]
  std::vector<float> feat_2;  // Intermediate activation for Stage 2 -> Stage 3: [B, 64, 16, 16]
  std::vector<float> feat_3;  // Intermediate activation for Stage 3 -> Stage 4: [B, 128, 8, 8]

  static std::size_t CalculateElementCount(int stage_index, std::int64_t batch_size);
  static std::size_t CalculateByteSize(int stage_index, std::int64_t batch_size);
  void ResizeForBatch(std::int64_t batch_size);
};

// A request-local trace. Counts are incremented only at the actual Session::Run
// call site (warmup is excluded); skipped stages have zero counts.
struct StageExecution {
  int stage_index = 0;
  std::vector<std::int64_t> input_shape;
  std::vector<std::vector<std::int64_t>> output_shapes;
  std::size_t copied_feature_elements = 0;
  std::vector<ExitEvaluation> sample_evaluations;
};

struct StagedInferenceResult {
  int selected_exit = 0;
  std::vector<float> logits;  // Owned [batch, 10], never a session arena pointer.
  std::vector<std::int64_t> output_shape;
  std::array<std::size_t, 4> stage_run_counts{};
  std::vector<StageExecution> stages;
};

/**
 * Persistent session manager and staged model runner for AeroNode.
 *
 * Maintains four persistent Ort::Session instances loaded once at boot time.
 * Recreating sessions per request is prohibited.
 */
class StagedInferenceEngine {
 public:
  explicit StagedInferenceEngine(const StagedEngineConfig& config);
  ~StagedInferenceEngine() = default;

  // Session, environment and buffer lifetimes stay at a stable address.
  StagedInferenceEngine(const StagedInferenceEngine&) = delete;
  StagedInferenceEngine& operator=(const StagedInferenceEngine&) = delete;
  StagedInferenceEngine(StagedInferenceEngine&&) = delete;
  StagedInferenceEngine& operator=(StagedInferenceEngine&&) = delete;

  // Serialized synchronous execution. The caller must keep input alive and
  // unchanged until return. State is copied once per request. The whole batch
  // continues if any sample's existing Phase-4 policy requests CONTINUE.
  StagedInferenceResult Infer(
      const std::vector<float>& input, const std::vector<std::int64_t>& shape,
      RuntimeState state, const ExitDecisionEngine& policy = ExitDecisionEngine{});

  // Execute warm-up forward pass on all 4 sessions to prime CPU caches & memory arenas
  void WarmUp();

  // Introspection accessors
  const StageMetadata& GetStageMetadata(std::size_t stage_index) const;
  const std::array<StageMetadata, 4>& GetAllStageMetadata() const noexcept { return metadata_; }
  bool IsInitialized() const noexcept { return is_initialized_; }
  bool IsWarmedUp() const noexcept { return is_warmed_up_; }

  // Legacy Phase-5C low-level access: externally synchronize these APIs. Never
  // resize a buffer or overlap raw session/buffer access with Infer/WarmUp while
  // an Ort::Value borrows it. Prefer Infer for safe owned results.
  IntermediateFallbackBuffers& GetFallbackBuffers() noexcept { return fallback_buffers_; }
  const IntermediateFallbackBuffers& GetFallbackBuffers() const noexcept { return fallback_buffers_; }

  // Create an Ort::Value input tensor wrapping the explicit fallback buffer
  Ort::Value CreateFallbackInputTensor(int stage_index, std::int64_t batch_size);

  // Raw session accessors (for testing and downstream execution)
  Ort::Session& GetSession(std::size_t stage_index);
  const Ort::Session& GetSession(std::size_t stage_index) const;

 private:
  void InitializeSessions();
  void InspectAndValidateContracts();
  std::vector<Ort::Value> RunStage(std::size_t stage_index, const Ort::Value& input,
                                 std::size_t* run_count = nullptr);

  StagedEngineConfig config_;
  Ort::Env environment_;
  Ort::SessionOptions session_options_;
  Ort::MemoryInfo memory_info_;
  std::array<std::unique_ptr<Ort::Session>, 4> sessions_;
  std::array<StageMetadata, 4> metadata_;
  IntermediateFallbackBuffers fallback_buffers_;
  std::mutex inference_mutex_;
  bool is_initialized_ = false;
  std::atomic<bool> is_warmed_up_{false};
};

}  // namespace aeronode

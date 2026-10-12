#include "staged_model_runner.hpp"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

void TestSuccessfulInitializationAndMetadata() {
  std::cout << "[RUN] TestSuccessfulInitializationAndMetadata\n";

  aeronode::StagedEngineConfig config;
  config.stage_model_paths = {
      "model/artifacts/phase5/stage_1.onnx",
      "model/artifacts/phase5/stage_2.onnx",
      "model/artifacts/phase5/stage_3.onnx",
      "model/artifacts/phase5/stage_4.onnx",
  };
  config.intra_op_threads = 1;

  aeronode::StagedInferenceEngine engine(config);
  assert(engine.IsInitialized());
  assert(!engine.IsWarmedUp());

  // Check metadata for Stage 1
  const auto& meta1 = engine.GetStageMetadata(0);
  assert(meta1.stage_index == 1);
  assert(meta1.stage_name == "stage_1");
  assert(meta1.inputs.size() == 1);
  assert(meta1.inputs[0].name == "input");
  assert(meta1.inputs[0].shape.size() == 4);
  assert(meta1.inputs[0].shape[1] == 3 && meta1.inputs[0].shape[2] == 128 && meta1.inputs[0].shape[3] == 128);
  assert(meta1.outputs.size() == 2);
  assert(meta1.outputs[0].name == "feat_1");
  assert(meta1.outputs[1].name == "exit_1_logits");

  // Check metadata for Stage 2
  const auto& meta2 = engine.GetStageMetadata(1);
  assert(meta2.stage_index == 2);
  assert(meta2.stage_name == "stage_2");
  assert(meta2.inputs.size() == 1);
  assert(meta2.inputs[0].name == "feat_1");
  assert(meta2.inputs[0].shape.size() == 4);
  assert(meta2.inputs[0].shape[1] == 32 && meta2.inputs[0].shape[2] == 32 && meta2.inputs[0].shape[3] == 32);
  assert(meta2.outputs.size() == 2);
  assert(meta2.outputs[0].name == "feat_2");
  assert(meta2.outputs[1].name == "exit_2_logits");

  // Check metadata for Stage 3
  const auto& meta3 = engine.GetStageMetadata(2);
  assert(meta3.stage_index == 3);
  assert(meta3.stage_name == "stage_3");
  assert(meta3.inputs.size() == 1);
  assert(meta3.inputs[0].name == "feat_2");
  assert(meta3.inputs[0].shape.size() == 4);
  assert(meta3.inputs[0].shape[1] == 64 && meta3.inputs[0].shape[2] == 16 && meta3.inputs[0].shape[3] == 16);
  assert(meta3.outputs.size() == 2);
  assert(meta3.outputs[0].name == "feat_3");
  assert(meta3.outputs[1].name == "exit_3_logits");

  // Check metadata for Stage 4
  const auto& meta4 = engine.GetStageMetadata(3);
  assert(meta4.stage_index == 4);
  assert(meta4.stage_name == "stage_4");
  assert(meta4.inputs.size() == 1);
  assert(meta4.inputs[0].name == "feat_3");
  assert(meta4.inputs[0].shape.size() == 4);
  assert(meta4.inputs[0].shape[1] == 128 && meta4.inputs[0].shape[2] == 8 && meta4.inputs[0].shape[3] == 8);
  assert(meta4.outputs.size() == 1);
  assert(meta4.outputs[0].name == "exit_4_logits");

  std::cout << "[PASS] TestSuccessfulInitializationAndMetadata\n";
}

void TestWarmUpSucceeds() {
  std::cout << "[RUN] TestWarmUpSucceeds\n";

  aeronode::StagedEngineConfig config;
  config.stage_model_paths = {
      "model/artifacts/phase5/stage_1.onnx",
      "model/artifacts/phase5/stage_2.onnx",
      "model/artifacts/phase5/stage_3.onnx",
      "model/artifacts/phase5/stage_4.onnx",
  };

  aeronode::StagedInferenceEngine engine(config);
  assert(!engine.IsWarmedUp());

  engine.WarmUp();
  assert(engine.IsWarmedUp());

  // Second warm-up call should also succeed without issue
  engine.WarmUp();
  assert(engine.IsWarmedUp());

  std::cout << "[PASS] TestWarmUpSucceeds\n";
}

void TestInvalidModelPathThrows() {
  std::cout << "[RUN] TestInvalidModelPathThrows\n";

  aeronode::StagedEngineConfig config;
  config.stage_model_paths = {
      "model/artifacts/phase5/stage_1.onnx",
      "non_existent_stage_2.onnx",
      "model/artifacts/phase5/stage_3.onnx",
      "model/artifacts/phase5/stage_4.onnx",
  };

  bool caught_exception = false;
  try {
    aeronode::StagedInferenceEngine engine(config);
  } catch (const std::exception& err) {
    caught_exception = true;
    std::string msg = err.what();
    assert(!msg.empty());
  }
  assert(caught_exception);

  std::cout << "[PASS] TestInvalidModelPathThrows\n";
}

void TestFallbackBufferCalculations() {
  std::cout << "[RUN] TestFallbackBufferCalculations\n";

  using Buffers = aeronode::IntermediateFallbackBuffers;

  // Batch size 1
  assert(Buffers::CalculateElementCount(1, 1) == 32768);
  assert(Buffers::CalculateByteSize(1, 1) == 32768 * sizeof(float));

  assert(Buffers::CalculateElementCount(2, 1) == 16384);
  assert(Buffers::CalculateByteSize(2, 1) == 16384 * sizeof(float));

  assert(Buffers::CalculateElementCount(3, 1) == 8192);
  assert(Buffers::CalculateByteSize(3, 1) == 8192 * sizeof(float));

  // Batch size 4
  assert(Buffers::CalculateElementCount(1, 4) == 4 * 32768);
  assert(Buffers::CalculateByteSize(1, 4) == 4 * 32768 * sizeof(float));

  assert(Buffers::CalculateElementCount(2, 4) == 4 * 16384);
  assert(Buffers::CalculateByteSize(2, 4) == 4 * 16384 * sizeof(float));

  assert(Buffers::CalculateElementCount(3, 4) == 4 * 8192);
  assert(Buffers::CalculateByteSize(3, 4) == 4 * 8192 * sizeof(float));

  // Invalid batch size <= 0 must throw
  bool caught_invalid_batch = false;
  try {
    Buffers::CalculateElementCount(1, 0);
  } catch (const std::invalid_argument&) {
    caught_invalid_batch = true;
  }
  assert(caught_invalid_batch);

  // Invalid stage index must throw
  bool caught_invalid_stage = false;
  try {
    Buffers::CalculateElementCount(4, 1);
  } catch (const std::invalid_argument&) {
    caught_invalid_stage = true;
  }
  assert(caught_invalid_stage);

  std::cout << "[PASS] TestFallbackBufferCalculations\n";
}

void TestFallbackTensorCreation() {
  std::cout << "[RUN] TestFallbackTensorCreation\n";

  aeronode::StagedEngineConfig config;
  config.stage_model_paths = {
      "model/artifacts/phase5/stage_1.onnx",
      "model/artifacts/phase5/stage_2.onnx",
      "model/artifacts/phase5/stage_3.onnx",
      "model/artifacts/phase5/stage_4.onnx",
  };

  aeronode::StagedInferenceEngine engine(config);

  // Create fallback tensor for stage 2 (consumes feat_1, batch 1)
  Ort::Value t2 = engine.CreateFallbackInputTensor(2, 1);
  assert(t2.IsTensor());
  assert(t2.GetTensorMutableData<float>() != nullptr);

  auto info2 = t2.GetTensorTypeAndShapeInfo();
  assert(info2.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  std::vector<std::int64_t> expected_s2 = {1, 32, 32, 32};
  assert(info2.GetShape() == expected_s2);
  assert(info2.GetElementCount() == 32768);

  // Modify buffer data through direct pointer and verify
  float* data2 = t2.GetTensorMutableData<float>();
  data2[0] = 42.0F;
  data2[32767] = -17.5F;
  assert(engine.GetFallbackBuffers().feat_1[0] == 42.0F);
  assert(engine.GetFallbackBuffers().feat_1[32767] == -17.5F);

  // Create fallback tensor for stage 3 (consumes feat_2, batch 2)
  Ort::Value t3 = engine.CreateFallbackInputTensor(3, 2);
  assert(t3.IsTensor());
  auto info3 = t3.GetTensorTypeAndShapeInfo();
  std::vector<std::int64_t> expected_s3 = {2, 64, 16, 16};
  assert(info3.GetShape() == expected_s3);
  assert(info3.GetElementCount() == 2 * 16384);

  // Create fallback tensor for stage 4 (consumes feat_3, batch 1)
  Ort::Value t4 = engine.CreateFallbackInputTensor(4, 1);
  assert(t4.IsTensor());
  auto info4 = t4.GetTensorTypeAndShapeInfo();
  std::vector<std::int64_t> expected_s4 = {1, 128, 8, 8};
  assert(info4.GetShape() == expected_s4);
  assert(info4.GetElementCount() == 8192);

  std::cout << "[PASS] TestFallbackTensorCreation\n";
}

}  // namespace

int main() {
  try {
    TestSuccessfulInitializationAndMetadata();
    TestWarmUpSucceeds();
    TestInvalidModelPathThrows();
    TestFallbackBufferCalculations();
    TestFallbackTensorCreation();
    std::cout << "\nAll 5 StagedInferenceEngine tests passed successfully.\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& err) {
    std::cerr << "Test failed with exception: " << err.what() << '\n';
    return EXIT_FAILURE;
  }
}

#include "entropy.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

void TestUniformDistribution() {
  std::vector<float> logits(10, 2.5F);  // All equal logits
  auto metrics = aeronode::ComputePredictionMetrics(logits);

  const float expected_entropy = std::log(10.0F);  // ~2.302585
  assert(std::abs(metrics.entropy - expected_entropy) < 1e-5F);
  assert(std::abs(metrics.confidence - 0.1F) < 1e-5F);
  assert(metrics.probabilities.size() == 10);

  float sum_p = 0.0F;
  for (float p : metrics.probabilities) {
    assert(std::abs(p - 0.1F) < 1e-5F);
    sum_p += p;
  }
  assert(std::abs(sum_p - 1.0F) < 1e-5F);
  std::cout << "[PASS] TestUniformDistribution\n";
}

void TestOneHotPeaked() {
  std::vector<float> logits(10, 0.0F);
  logits[3] = 100.0F;  // Strongly peaked at class 3
  auto metrics = aeronode::ComputePredictionMetrics(logits);

  assert(metrics.predicted_class == 3);
  assert(std::abs(metrics.confidence - 1.0F) < 1e-5F);
  assert(std::abs(metrics.entropy - 0.0F) < 1e-5F);
  std::cout << "[PASS] TestOneHotPeaked\n";
}

void TestLargeDynamicRange() {
  // Extreme positive values should not overflow exp()
  std::vector<float> logits = {10000.0F, 9999.0F, 9998.0F, 0.0F, -1000.0F};
  auto metrics = aeronode::ComputePredictionMetrics(logits);

  assert(metrics.predicted_class == 0);
  assert(!std::isnan(metrics.entropy));
  assert(!std::isinf(metrics.entropy));
  assert(!std::isnan(metrics.confidence));
  assert(metrics.confidence > 0.6F);
  std::cout << "[PASS] TestLargeDynamicRange\n";
}

void TestExtremeNegative() {
  // Extreme negative equal values
  std::vector<float> logits(10, -5000.0F);
  auto metrics = aeronode::ComputePredictionMetrics(logits);

  const float expected_entropy = std::log(10.0F);
  assert(std::abs(metrics.entropy - expected_entropy) < 1e-4F);
  assert(!std::isnan(metrics.entropy));
  assert(!std::isnan(metrics.confidence));
  std::cout << "[PASS] TestExtremeNegative\n";
}

void TestArbitraryKnownDistribution() {
  // Logits: [1.0, 2.0, 3.0]
  // exp: [e^1, e^2, e^3] = [2.7182818, 7.389056, 20.085537] -> sum = 30.192875
  // p: [0.09003057, 0.24472847, 0.66524096]
  // H = -(0.09003057*ln(0.09003057) + 0.24472847*ln(0.24472847) + 0.66524096*ln(0.66524096))
  //   = -(-0.216743 + -0.344498 + -0.271142) = 0.832383
  std::vector<float> logits = {1.0F, 2.0F, 3.0F};
  auto metrics = aeronode::ComputePredictionMetrics(logits);

  assert(metrics.predicted_class == 2);
  assert(std::abs(metrics.confidence - 0.66524096F) < 1e-4F);
  assert(std::abs(metrics.entropy - 0.832383F) < 1e-4F);
  std::cout << "[PASS] TestArbitraryKnownDistribution\n";
}

int main() {
  try {
    TestUniformDistribution();
    TestOneHotPeaked();
    TestLargeDynamicRange();
    TestExtremeNegative();
    TestArbitraryKnownDistribution();
    std::cout << "\nAll 5 entropy tests passed successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "Test failed: " << ex.what() << '\n';
    return 1;
  }
}

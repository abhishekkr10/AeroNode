#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace aeronode {

struct PredictionMetrics {
  int predicted_class = 0;
  float confidence = 0.0F;     // Maximum class probability in [0.0, 1.0]
  float entropy = 0.0F;        // Shannon entropy in nats: -sum(p * ln(p))
  std::vector<float> probabilities;
};

/**
 * Computes numerically stable softmax, predicted class, confidence,
 * and Shannon entropy (in natural units / nats) from raw logits.
 *
 * Uses log-sum-exp formulation for entropy:
 *   H(p) = ln(S) - sum(p_i * (z_i - max(z)))
 * where S = sum(exp(z_i - max(z))).
 * This avoids underflow to 0 when evaluating ln(p_i).
 */
inline PredictionMetrics ComputePredictionMetrics(const float* logits, std::size_t num_classes) {
  if (logits == nullptr || num_classes == 0) {
    throw std::invalid_argument("ComputePredictionMetrics requires non-null logits and num_classes > 0");
  }

  PredictionMetrics metrics;
  metrics.probabilities.resize(num_classes);

  // 1. Find maximum logit for numerical stability
  float max_logit = logits[0];
  std::size_t best_class = 0;
  for (std::size_t i = 1; i < num_classes; ++i) {
    if (logits[i] > max_logit) {
      max_logit = logits[i];
      best_class = i;
    }
  }
  metrics.predicted_class = static_cast<int>(best_class);

  // 2. Compute exp(z_i - max_logit) and sum S in double precision
  std::vector<double> exp_vals(num_classes);
  double sum_exp = 0.0;
  for (std::size_t i = 0; i < num_classes; ++i) {
    exp_vals[i] = std::exp(static_cast<double>(logits[i] - max_logit));
    sum_exp += exp_vals[i];
  }

  // 3. Normalize probabilities and calculate entropy
  // H = ln(S) - sum(p_i * (z_i - max_logit))
  const double ln_sum_exp = std::log(sum_exp);
  double p_dot_shifted = 0.0;
  float max_prob = 0.0F;

  for (std::size_t i = 0; i < num_classes; ++i) {
    const double p_i = exp_vals[i] / sum_exp;
    metrics.probabilities[i] = static_cast<float>(p_i);
    if (metrics.probabilities[i] > max_prob) {
      max_prob = metrics.probabilities[i];
    }
    p_dot_shifted += p_i * static_cast<double>(logits[i] - max_logit);
  }

  metrics.confidence = max_prob;
  double entropy = ln_sum_exp - p_dot_shifted;
  if (entropy < 0.0) {
    entropy = 0.0;  // Guard against tiny negative floating-point epsilon
  }
  metrics.entropy = static_cast<float>(entropy);

  return metrics;
}

inline PredictionMetrics ComputePredictionMetrics(const std::vector<float>& logits) {
  return ComputePredictionMetrics(logits.data(), logits.size());
}

}  // namespace aeronode

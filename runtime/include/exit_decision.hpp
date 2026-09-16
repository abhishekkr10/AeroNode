#pragma once

#include "entropy.hpp"
#include "runtime_state.hpp"

#include <array>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace aeronode {

enum class DecisionReason {
  CONFIDENCE_MET,
  LATENCY_BUDGET_EXHAUSTED,
  RESOURCE_PRESSURE_FORCED_EXIT,
  FINAL_EXIT_REACHED
};

inline std::string ToString(DecisionReason reason) {
  switch (reason) {
    case DecisionReason::CONFIDENCE_MET:
      return "CONFIDENCE_MET";
    case DecisionReason::LATENCY_BUDGET_EXHAUSTED:
      return "LATENCY_BUDGET_EXHAUSTED";
    case DecisionReason::RESOURCE_PRESSURE_FORCED_EXIT:
      return "RESOURCE_PRESSURE_FORCED_EXIT";
    case DecisionReason::FINAL_EXIT_REACHED:
      return "FINAL_EXIT_REACHED";
  }
  return "UNKNOWN";
}

struct ExitEvaluation {
  int exit_index = 0;              // 1 to 4
  std::string exit_name;
  float entropy = 0.0F;
  float confidence = 0.0F;
  float threshold_used = 0.0F;
  int predicted_class = 0;
  bool exited = false;
  std::string action;              // "EXIT" or "CONTINUE"
  DecisionReason reason = DecisionReason::FINAL_EXIT_REACHED;
};

struct DecisionResult {
  int selected_exit = 4;           // 1 to 4
  std::string selected_exit_name;
  int predicted_class = 0;
  float confidence = 0.0F;
  float entropy = 0.0F;
  DecisionReason reason = DecisionReason::FINAL_EXIT_REACHED;
  std::vector<ExitEvaluation> evaluations;
};

struct ExitPolicyConfig {
  // Baseline entropy thresholds for exits 1, 2, 3, 4 (in nats)
  // Lower entropy means higher confidence.
  std::array<float, 4> entropy_thresholds = {0.60F, 0.80F, 1.10F, 2.50F};
  float pressure_threshold = 0.80F;          // Above this, trigger aggressive exit
  float pressure_entropy_slack = 0.30F;      // Relaxed entropy allowance under pressure
  float stage_latency_estimate_ms = 3.0F;    // Estimated ms per stage for deadline projection
};

class ExitDecisionEngine {
 public:
  explicit ExitDecisionEngine(ExitPolicyConfig config = ExitPolicyConfig{})
      : config_(config) {}

  DecisionResult Evaluate(
      const std::vector<PredictionMetrics>& exit_metrics,
      const RuntimeState& state,
      double elapsed_ms = 0.0) const {
    const std::size_t num_exits = exit_metrics.size();
    if (num_exits == 0) {
      throw std::invalid_argument("ExitDecisionEngine requires at least one exit metrics");
    }

    DecisionResult result;
    result.evaluations.reserve(num_exits);

    const std::vector<std::string> exit_names = {
        "exit_1_logits", "exit_2_logits", "exit_3_logits", "exit_4_logits"};

    for (std::size_t i = 0; i < num_exits; ++i) {
      const auto& metrics = exit_metrics[i];
      const int exit_idx = static_cast<int>(i + 1);
      const std::string exit_name = (i < exit_names.size()) ? exit_names[i] : ("exit_" + std::to_string(exit_idx));
      const float base_threshold = (i < config_.entropy_thresholds.size()) ? config_.entropy_thresholds[i] : 2.0F;

      ExitEvaluation eval;
      eval.exit_index = exit_idx;
      eval.exit_name = exit_name;
      eval.entropy = metrics.entropy;
      eval.confidence = metrics.confidence;
      eval.predicted_class = metrics.predicted_class;
      eval.threshold_used = base_threshold;

      // Check if this is the final exit
      if (i == num_exits - 1) {
        eval.exited = true;
        eval.action = "EXIT";
        eval.reason = DecisionReason::FINAL_EXIT_REACHED;
        result.evaluations.push_back(eval);

        result.selected_exit = exit_idx;
        result.selected_exit_name = exit_name;
        result.predicted_class = metrics.predicted_class;
        result.confidence = metrics.confidence;
        result.entropy = metrics.entropy;
        result.reason = eval.reason;
        break;
      }

      // Condition 1: High confidence (entropy <= threshold)
      if (metrics.entropy <= base_threshold) {
        eval.exited = true;
        eval.action = "EXIT";
        eval.reason = DecisionReason::CONFIDENCE_MET;
        result.evaluations.push_back(eval);

        result.selected_exit = exit_idx;
        result.selected_exit_name = exit_name;
        result.predicted_class = metrics.predicted_class;
        result.confidence = metrics.confidence;
        result.entropy = metrics.entropy;
        result.reason = eval.reason;
        break;
      }

      // Condition 2: Latency budget exhausted or projected to exceed budget
      const double projected_next_ms = elapsed_ms + static_cast<double>(config_.stage_latency_estimate_ms);
      if (projected_next_ms >= static_cast<double>(state.latency_budget_ms)) {
        eval.exited = true;
        eval.action = "EXIT";
        eval.reason = DecisionReason::LATENCY_BUDGET_EXHAUSTED;
        result.evaluations.push_back(eval);

        result.selected_exit = exit_idx;
        result.selected_exit_name = exit_name;
        result.predicted_class = metrics.predicted_class;
        result.confidence = metrics.confidence;
        result.entropy = metrics.entropy;
        result.reason = eval.reason;
        break;
      }

      // Condition 3: High resource pressure with acceptable confidence
      if (state.resource_pressure >= config_.pressure_threshold &&
          metrics.entropy <= (base_threshold + config_.pressure_entropy_slack)) {
        eval.exited = true;
        eval.action = "EXIT";
        eval.reason = DecisionReason::RESOURCE_PRESSURE_FORCED_EXIT;
        result.evaluations.push_back(eval);

        result.selected_exit = exit_idx;
        result.selected_exit_name = exit_name;
        result.predicted_class = metrics.predicted_class;
        result.confidence = metrics.confidence;
        result.entropy = metrics.entropy;
        result.reason = eval.reason;
        break;
      }

      // Default: Continue to next exit
      eval.exited = false;
      eval.action = "CONTINUE";
      result.evaluations.push_back(eval);
    }

    return result;
  }

 private:
  ExitPolicyConfig config_;
};

}  // namespace aeronode

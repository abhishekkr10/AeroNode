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
  std::string source_checkpoint_sha256;     // Optional binding for calibrated artifacts.
};

class ExitDecisionEngine {
 public:
  explicit ExitDecisionEngine(ExitPolicyConfig config = ExitPolicyConfig{})
      : config_(config) {}

  const std::string& SourceCheckpointSha256() const noexcept { return config_.source_checkpoint_sha256; }

  // Incremental evaluation for the four-stage execution engine. An available
  // intermediate head is not the final head merely because later logits have
  // not been computed yet. Rules and priority are shared with Evaluate below.
  ExitEvaluation EvaluateStage(const PredictionMetrics& metrics, int exit_index,
                               const RuntimeState& state, double elapsed_ms = 0.0) const {
    if (exit_index < 1 || exit_index > 4) {
      throw std::invalid_argument("EvaluateStage requires exit_index in [1, 4]");
    }
    return EvaluateAt(metrics, static_cast<std::size_t>(exit_index - 1),
                      exit_index == 4, state, elapsed_ms);
  }

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

    for (std::size_t i = 0; i < num_exits; ++i) {
      const auto eval = EvaluateAt(exit_metrics[i], i, i == num_exits - 1, state, elapsed_ms);
      result.evaluations.push_back(eval);
      if (eval.exited) {
        result.selected_exit = eval.exit_index;
        result.selected_exit_name = eval.exit_name;
        result.predicted_class = eval.predicted_class;
        result.confidence = eval.confidence;
        result.entropy = eval.entropy;
        result.reason = eval.reason;
        break;
      }
    }

    return result;
  }

 private:
  ExitEvaluation EvaluateAt(const PredictionMetrics& metrics, std::size_t i,
                            bool final, const RuntimeState& state, double elapsed_ms) const {
    ExitEvaluation eval;
    eval.exit_index = static_cast<int>(i + 1);
    eval.exit_name = "exit_" + std::to_string(i + 1) + (i < 4 ? "_logits" : "");
    eval.entropy = metrics.entropy;
    eval.confidence = metrics.confidence;
    eval.predicted_class = metrics.predicted_class;
    eval.threshold_used = i < config_.entropy_thresholds.size() ? config_.entropy_thresholds[i] : 2.0F;

    if (final) {
      eval.exited = true;
      eval.reason = DecisionReason::FINAL_EXIT_REACHED;
    } else if (metrics.entropy <= eval.threshold_used) {
      eval.exited = true;
      eval.reason = DecisionReason::CONFIDENCE_MET;
    } else if (elapsed_ms + static_cast<double>(config_.stage_latency_estimate_ms) >=
               static_cast<double>(state.latency_budget_ms)) {
      eval.exited = true;
      eval.reason = DecisionReason::LATENCY_BUDGET_EXHAUSTED;
    } else if (state.resource_pressure >= config_.pressure_threshold &&
               metrics.entropy <= eval.threshold_used + config_.pressure_entropy_slack) {
      eval.exited = true;
      eval.reason = DecisionReason::RESOURCE_PRESSURE_FORCED_EXIT;
    }
    eval.action = eval.exited ? "EXIT" : "CONTINUE";
    return eval;
  }

  ExitPolicyConfig config_;
};

}  // namespace aeronode

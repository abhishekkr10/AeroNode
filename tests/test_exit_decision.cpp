#include "exit_decision.hpp"

#include <cassert>
#include <iostream>
#include <vector>

void TestConfidenceExit() {
  aeronode::ExitDecisionEngine engine;
  aeronode::RuntimeState state;
  state.latency_budget_ms = 50.0F;
  state.resource_pressure = 0.2F;

  // Exit 1 has low entropy (0.35 <= 0.60)
  std::vector<aeronode::PredictionMetrics> metrics(4);
  metrics[0].entropy = 0.35F;
  metrics[0].confidence = 0.90F;
  metrics[0].predicted_class = 2;

  metrics[1].entropy = 0.50F;
  metrics[2].entropy = 0.40F;
  metrics[3].entropy = 0.30F;

  auto result = engine.Evaluate(metrics, state, 1.0);
  assert(result.selected_exit == 1);
  assert(result.selected_exit_name == "exit_1_logits");
  assert(result.reason == aeronode::DecisionReason::CONFIDENCE_MET);
  assert(result.evaluations.size() == 1);
  assert(result.evaluations[0].action == "EXIT");
  std::cout << "[PASS] TestConfidenceExit\n";
}

void TestLatencyBudgetExit() {
  aeronode::ExitDecisionEngine engine;
  aeronode::RuntimeState state;
  state.latency_budget_ms = 3.5F;  // Very tight budget
  state.resource_pressure = 0.3F;

  // Exit 1 has high entropy (1.8 > 0.60), but projected time (1.0 + 3.0 = 4.0ms) > 3.5ms
  std::vector<aeronode::PredictionMetrics> metrics(4);
  metrics[0].entropy = 1.8F;
  metrics[0].confidence = 0.4F;
  metrics[0].predicted_class = 5;

  auto result = engine.Evaluate(metrics, state, 1.0);
  assert(result.selected_exit == 1);
  assert(result.reason == aeronode::DecisionReason::LATENCY_BUDGET_EXHAUSTED);
  std::cout << "[PASS] TestLatencyBudgetExit\n";
}

void TestResourcePressureForcedExit() {
  aeronode::ExitDecisionEngine engine;
  aeronode::RuntimeState state;
  state.latency_budget_ms = 50.0F;
  state.resource_pressure = 0.85F;  // High pressure

  // Exit 1: entropy 1.5 > 0.6 + 0.3 (continues)
  // Exit 2: entropy 0.95 <= 0.80 + 0.30 = 1.10 (triggers forced exit under pressure)
  std::vector<aeronode::PredictionMetrics> metrics(4);
  metrics[0].entropy = 1.5F;
  metrics[0].confidence = 0.35F;

  metrics[1].entropy = 0.95F;
  metrics[1].confidence = 0.65F;
  metrics[1].predicted_class = 7;

  metrics[2].entropy = 0.8F;
  metrics[3].entropy = 0.5F;

  auto result = engine.Evaluate(metrics, state, 1.0);
  assert(result.selected_exit == 2);
  assert(result.selected_exit_name == "exit_2_logits");
  assert(result.reason == aeronode::DecisionReason::RESOURCE_PRESSURE_FORCED_EXIT);
  assert(result.evaluations.size() == 2);
  assert(result.evaluations[0].action == "CONTINUE");
  assert(result.evaluations[1].action == "EXIT");
  std::cout << "[PASS] TestResourcePressureForcedExit\n";
}

void TestFinalExitReached() {
  aeronode::ExitDecisionEngine engine;
  aeronode::RuntimeState state;
  state.latency_budget_ms = 100.0F;
  state.resource_pressure = 0.1F;

  // All intermediate exits have high entropy and generous resources
  std::vector<aeronode::PredictionMetrics> metrics(4);
  metrics[0].entropy = 2.0F;
  metrics[1].entropy = 1.8F;
  metrics[2].entropy = 1.5F;
  metrics[3].entropy = 0.9F;
  metrics[3].confidence = 0.70F;
  metrics[3].predicted_class = 3;

  auto result = engine.Evaluate(metrics, state, 1.0);
  assert(result.selected_exit == 4);
  assert(result.selected_exit_name == "exit_4_logits");
  assert(result.reason == aeronode::DecisionReason::FINAL_EXIT_REACHED);
  assert(result.evaluations.size() == 4);
  assert(result.evaluations[0].action == "CONTINUE");
  assert(result.evaluations[1].action == "CONTINUE");
  assert(result.evaluations[2].action == "CONTINUE");
  assert(result.evaluations[3].action == "EXIT");
  std::cout << "[PASS] TestFinalExitReached\n";
}

int main() {
  try {
    TestConfidenceExit();
    TestLatencyBudgetExit();
    TestResourcePressureForcedExit();
    TestFinalExitReached();
    std::cout << "\nAll 4 ExitDecision tests passed successfully.\n";
    return 0;
  } catch (const std::exception& ex) {
    std::cerr << "ExitDecision test failed: " << ex.what() << '\n';
    return 1;
  }
}

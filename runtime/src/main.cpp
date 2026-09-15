#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Options {
  std::string model_path;
  int iterations = 100;
};

Options ParseOptions(int argc, char* argv[]) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--model" && index + 1 < argc) {
      options.model_path = argv[++index];
    } else if (argument == "--iterations" && index + 1 < argc) {
      options.iterations = std::stoi(argv[++index]);
    } else {
      throw std::invalid_argument("Usage: aeronode_runtime --model <path> [--iterations N]");
    }
  }
  if (options.model_path.empty() || options.iterations <= 0) {
    throw std::invalid_argument("Usage: aeronode_runtime --model <path> [--iterations N]");
  }
  return options;
}

double Percentile95(std::vector<double> samples) {
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

}  // namespace

int main(int argc, char* argv[]) {
  try {
    const Options options = ParseOptions(argc, argv);
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
    const auto input_name = session.GetInputNameAllocated(0, allocator);
    const auto output_name = session.GetOutputNameAllocated(0, allocator);
    const auto input_info = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const auto output_info = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo();
    const std::vector<std::int64_t> input_shape = input_info.GetShape();
    const std::vector<std::int64_t> output_shape = output_info.GetShape();
    if (input_shape != std::vector<std::int64_t>({1, 4}) ||
        output_shape != std::vector<std::int64_t>({1, 3})) {
      throw std::runtime_error("Unexpected model tensor shapes; expected input [1,4] and output [1,3]");
    }

    std::array<float, 4> input_data{-1.0F, 0.5F, 2.0F, -0.25F};
    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info, input_data.data(), input_data.size(), input_shape.data(), input_shape.size());
    const std::array<const char*, 1> input_names{input_name.get()};
    const std::array<const char*, 1> output_names{output_name.get()};

    auto run_once = [&]() {
      return session.Run(Ort::RunOptions{nullptr}, input_names.data(), &input_tensor, 1,
                         output_names.data(), 1);
    };

    run_once();  // Warm-up is intentionally excluded from measurements.
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
    const auto predicted_class = static_cast<int>(std::distance(logits, std::max_element(logits, logits + 3)));
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
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "aeronode_runtime: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}

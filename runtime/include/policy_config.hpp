#pragma once

#include "exit_decision.hpp"

#include <cmath>
#include <fstream>
#include <filesystem>
#include <stdexcept>

namespace aeronode {

// Optional externally calibrated thresholds. Existing defaults and rules remain
// unchanged unless a caller explicitly supplies this versioned configuration.
inline ExitPolicyConfig LoadExitPolicy(const std::string& path) {
  try {
    std::ifstream file(std::filesystem::u8path(path));
    if (!file) throw std::invalid_argument("cannot open file");
    nlohmann::json json;
    file >> json;
    if (json.at("schema_version") != 1 || !json.at("entropy_thresholds").is_array() ||
        json.at("entropy_thresholds").size() != 4) {
      throw std::invalid_argument("requires schema_version 1 and four entropy thresholds");
    }
    ExitPolicyConfig config;
    if (json.contains("source_checkpoint_sha256")) {
      config.source_checkpoint_sha256 = json.at("source_checkpoint_sha256").get<std::string>();
      if (config.source_checkpoint_sha256.size() != 64 ||
          config.source_checkpoint_sha256.find_first_not_of("0123456789abcdefABCDEF") != std::string::npos) {
        throw std::invalid_argument("invalid checkpoint hash");
      }
    }
    for (std::size_t i = 0; i < 4; ++i) {
      const auto& value = json.at("entropy_thresholds")[i];
      if (!value.is_number()) throw std::invalid_argument("threshold must be numeric");
      config.entropy_thresholds[i] = value.get<float>();
      if (!std::isfinite(config.entropy_thresholds[i])) throw std::invalid_argument("threshold must be finite");
    }
    return config;
  } catch (const std::exception& error) {
    throw std::invalid_argument("Invalid exit policy '" + path + "': " + error.what());
  }
}

}  // namespace aeronode

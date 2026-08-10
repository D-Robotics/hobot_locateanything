#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace locateanything {

struct GraphTiming {
  std::string graph;
  int32_t calls = 0;
  double total_ms = 0.0;
  double bpu_wait_ms = 0.0;
  double submit_ms = 0.0;
  uint64_t input_bytes = 0;
  uint64_t output_bytes = 0;
};

struct LanguageMetrics {
  int32_t prompt_tokens = 0;
  int32_t generated_tokens = 0;
  int32_t pbd_calls = 0;
  int32_t pbd_accepted_tokens = 0;
  int32_t ar_calls = 0;
  int32_t ar_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double cache_update_ms = 0.0;
  double host_decode_ms = 0.0;
  std::vector<GraphTiming> graph_timings;
  std::string executed_mode;
  std::string fallback_reason;
};

struct StageTiming {
  // Monotonic offsets from the start of one InferenceSession::Infer call.
  // Consumers can map these offsets onto their own wall/ROS clock.
  double start_ms = 0.0;
  double end_ms = 0.0;

  double DurationMs() const { return end_ms > start_ms ? end_ms - start_ms : 0.0; }
};

struct InferenceMetrics {
  double preprocess_ms = 0.0;
  double vision_ms = 0.0;
  double language_ms = 0.0;
  double postprocess_ms = 0.0;
  double total_ms = 0.0;
  StageTiming preprocess_timing;
  StageTiming vision_timing;
  StageTiming language_timing;
  StageTiming postprocess_timing;
  LanguageMetrics language;
};

}  // namespace locateanything

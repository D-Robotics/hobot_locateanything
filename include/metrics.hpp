#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace locateanything {

/** Per-graph execution counters collected by the HBM wrapper. */
struct GraphTiming {
  std::string graph;
  int32_t calls = 0;
  double total_ms = 0.0;
  double input_build_ms = 0.0;
  double buffer_prepare_ms = 0.0;
  double input_pack_ms = 0.0;
  double input_flush_ms = 0.0;
  double bpu_wait_ms = 0.0;
  double submit_ms = 0.0;
  double output_flush_ms = 0.0;
  double output_unpack_ms = 0.0;
  uint64_t input_bytes = 0;
  uint64_t resident_input_bytes = 0;
  uint64_t output_bytes = 0;
};

/** Count of one decoder decision or token-transition category. */
struct DecodeEventCount {
  std::string event;
  int32_t count = 0;
};

/** Host-side Language generation counters and fallback information. */
struct LanguageMetrics {
  int32_t prompt_tokens = 0;
  int32_t generated_tokens = 0;
  int32_t pbd_calls = 0;
  int32_t pbd_accepted_tokens = 0;
  int32_t ar_calls = 0;
  int32_t ar_tokens = 0;
  double prefill_ms = 0.0;
  double decode_ms = 0.0;
  double cache_initialize_ms = 0.0;
  double cache_seed_ms = 0.0;
  double cache_update_ms = 0.0;
  double host_decode_ms = 0.0;
  std::vector<GraphTiming> graph_timings;
  std::vector<DecodeEventCount> decode_events;
  std::string executed_mode;
  std::string fallback_reason;
};

/** Monotonic start/end offsets for one pipeline stage. */
struct StageTiming {
  // Monotonic offsets from the start of one InferenceSession::Infer call.
  // Consumers can map these offsets onto their own wall/ROS clock.
  double start_ms = 0.0;
  double end_ms = 0.0;

  /**
   * @brief Return the non-negative duration represented by this interval.
   * @return Stage duration in milliseconds.
   */
  double DurationMs() const { return end_ms > start_ms ? end_ms - start_ms : 0.0; }
};

/** End-to-end timings for one inference call. */
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

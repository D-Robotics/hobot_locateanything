// LocateAnything Language HBM runner.
//
// Executes the compiled fixed graphs directly:
//   prefill (profile q) -> PBD (q=6..12) / AR (q=1..5)
// It supports both Hybrid PBD and full autoregressive generation.

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <initializer_list>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "runtime/attention_mask.hpp"
#include "runtime/decoder.hpp"
#include "runtime/embedding.hpp"
#include "runtime/hbm.hpp"
#include "runtime/kv_cache.hpp"
#include "runtime/language.hpp"
#include "runtime/language_graphs.hpp"

namespace rt = locateanything_runtime;

namespace {

constexpr int32_t kF16 = 4;
constexpr int32_t kS32 = 8;
constexpr int32_t kVocab = 152681;
constexpr int32_t kHidden = 2048;
constexpr int32_t kCacheCount = 72;
constexpr int32_t kImageToken = 151665;
constexpr int32_t kTextMaskToken = 151676;
constexpr int32_t kBoxStartToken = 151668;
constexpr int32_t kBoxEndToken = 151669;
constexpr int32_t kImEndToken = 151645;
constexpr int32_t kNoneToken = 4064;
constexpr uint16_t kMaskValue = 0xf800;  // fp16(-32768.0)

/**
 * @brief Multiply tensor dimensions into an element count.
 * @param shape Tensor dimensions.
 * @return Number of logical tensor elements.
 */
int64_t ElementCount(const std::vector<int32_t>& shape) {
  int64_t count = 1;
  for (int32_t value : shape) count *= value;
  return count;
}

/**
 * @brief Compare a runtime shape with a fixed expected shape.
 * @param left Runtime tensor shape.
 * @param right Expected dimensions.
 * @return True when every dimension matches.
 */
bool SameShape(const std::vector<int32_t>& left,
               std::initializer_list<int32_t> right) {
  return left == std::vector<int32_t>(right);
}

struct LanguageAbiProfile {
  bool fused_prefill = false;
  bool compact_logits = false;
  int32_t prefill_len = 0;
  int32_t cache_len = 0;
};

/** Return the fixed query length encoded by one Language graph name. */
int32_t LanguageGraphQueryLength(const std::string& name,
                                 int32_t prefill_len) {
  if (name == "prefill") return prefill_len;
  if (name == "decode") return 6;
  if (name == "decode_ar") return 1;
  for (const std::string prefix : {"decode_pbd_q", "decode_ar_q"}) {
    if (name.rfind(prefix, 0) != 0) continue;
    const std::string suffix = name.substr(prefix.size());
    if (suffix.empty() || !std::all_of(
                              suffix.begin(), suffix.end(),
                              [](unsigned char value) {
                                return std::isdigit(value) != 0;
                              })) {
      return -1;
    }
    return std::stoi(suffix);
  }
  return -1;
}

/** Return true for q6-q12 PBD graph names. */
bool IsPbdGraph(const std::string& name) {
  return name == "decode" || name.rfind("decode_pbd_q", 0) == 0;
}

/** Validate graph shapes shared by legacy and fused/compact Language HBMs. */
LanguageAbiProfile ValidateLanguageAbi(rt::HbmSession* session) {
  if (session == nullptr) {
    throw std::runtime_error("Language HBM session is null");
  }
  rt::Graph* prefill = session->GetGraph("prefill");
  if (prefill == nullptr || prefill->GetInputShapes().size() != 75 ||
      prefill->GetOutputShapes().size() != 73) {
    throw std::runtime_error(
        "Language HBM Prefill must expose 75 inputs and 73 outputs");
  }
  const auto& prefill_inputs = prefill->GetInputShapes();
  const auto& prefill_outputs = prefill->GetOutputShapes();
  if (prefill_inputs[0].size() != 3 || prefill_inputs[0][0] != 1 ||
      prefill_inputs[0][2] != kHidden || prefill_inputs[2].size() != 3 ||
      prefill_inputs[2][0] != 1 || prefill_outputs[0].size() != 3 ||
      prefill_outputs[0][0] != 1 || prefill_outputs[0][2] != kVocab) {
    throw std::runtime_error("Language HBM Prefill has an invalid base ABI");
  }
  LanguageAbiProfile profile;
  profile.prefill_len = prefill_inputs[0][1];
  profile.cache_len = prefill_inputs[2][2];
  const int32_t prefill_logits_rows = prefill_outputs[0][1];
  if (prefill_logits_rows == 7) {
    profile.fused_prefill = true;
  } else if (prefill_logits_rows != 1) {
    throw std::runtime_error(
        "Language HBM Prefill logits must contain either 1 legacy row or "
        "7 fused rows");
  }
  if (profile.prefill_len < 128 || profile.cache_len <= profile.prefill_len) {
    throw std::runtime_error("Language HBM Prefill/cache capacity is invalid");
  }
  rt::Graph* pbd_q7 = session->GetGraph("decode_pbd_q7");
  rt::Graph* ar_q2 = session->GetGraph("decode_ar_q2");
  if (pbd_q7 == nullptr || ar_q2 == nullptr ||
      pbd_q7->GetOutputShapes().empty() || ar_q2->GetOutputShapes().empty() ||
      pbd_q7->GetOutputShapes()[0].size() != 3 ||
      ar_q2->GetOutputShapes()[0].size() != 3) {
    throw std::runtime_error("Language HBM cannot identify Compact Logits ABI");
  }
  const int32_t pbd_q7_rows = pbd_q7->GetOutputShapes()[0][1];
  const int32_t ar_q2_rows = ar_q2->GetOutputShapes()[0][1];
  if (pbd_q7_rows == 6 && ar_q2_rows == 1) {
    profile.compact_logits = true;
  } else if (pbd_q7_rows != 7 || ar_q2_rows != 2) {
    throw std::runtime_error(
        "Language HBM mixes compact and full-query Decode logits");
  }

  for (const std::string& name : rt::LanguageGraphNames()) {
    rt::Graph* graph = session->GetGraph(name);
    if (graph == nullptr) {
      throw std::runtime_error("Language HBM graph is missing: " + name);
    }
    const auto& inputs = graph->GetInputShapes();
    const auto& outputs = graph->GetOutputShapes();
    const auto& input_dtypes = graph->GetInputDtypes();
    const auto& output_dtypes = graph->GetOutputDtypes();
    const int32_t query = LanguageGraphQueryLength(name, profile.prefill_len);
    if (query <= 0 || inputs.size() != 75 || outputs.size() != 73 ||
        input_dtypes.size() != inputs.size() ||
        output_dtypes.size() != outputs.size()) {
      throw std::runtime_error("Language HBM graph count ABI mismatch: " + name);
    }
    const int32_t expected_logits_rows =
        name == "prefill"
            ? (profile.fused_prefill ? 7 : 1)
            : (profile.compact_logits ? (IsPbdGraph(name) ? 6 : 1) : query);
    if (!SameShape(inputs[0], {1, query, kHidden}) ||
        !SameShape(inputs[1], {1, 1, query}) ||
        !SameShape(inputs[2], {1, query, profile.cache_len}) ||
        !SameShape(outputs[0], {1, expected_logits_rows, kVocab}) ||
        input_dtypes[0] != kF16 || input_dtypes[1] != kS32 ||
        input_dtypes[2] != kF16 || output_dtypes[0] != kF16) {
      throw std::runtime_error("Language HBM primary tensor ABI mismatch: " +
                               name);
    }
    for (int32_t index = 0; index < kCacheCount; ++index) {
      const size_t input_index = 3 + static_cast<size_t>(index);
      const size_t output_index = 1 + static_cast<size_t>(index);
      if (!SameShape(inputs[input_index],
                     {1, profile.cache_len, 2, 128}) ||
          !SameShape(outputs[output_index], {1, query, 2, 128}) ||
          input_dtypes[input_index] != output_dtypes[output_index]) {
        throw std::runtime_error("Language HBM KV ABI mismatch: " + name);
      }
    }
  }
  return profile;
}

/**
 * @brief Encode one host float as an IEEE-754 binary16 bit pattern.
 * @param value Host floating-point value.
 * @return Raw fp16 bits consumed by the HBM graph.
 */
uint16_t FloatToFp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0) return static_cast<uint16_t>(sign);
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  mantissa = (mantissa + 0x1000u) >> 13;
  if (mantissa == 0x400u) {
    mantissa = 0;
    if (++exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               mantissa);
}

struct CacheState {
  std::vector<rt::Tensor> tensors;
};

struct PreparedInputs {
  rt::Tensor embeddings;
  rt::Tensor positions;
  rt::Tensor mask;
  rt::Tensor history_mask;
  rt::Tensor random_values;
  std::vector<const rt::Tensor*> views;
};

struct EngineState {
  rt::HbmSession session;
  rt::EmbedLookup embed;
  PreparedInputs prepared_inputs;
  CacheState prefill_cache;
  CacheState full_cache;
  std::vector<rt::Tensor> prefill_outputs;
  std::vector<rt::Tensor> pbd_outputs;
  // Language graphs run serially; retain transition/output storage between
  // calls instead of allocating a new vector for every AR step.
  std::vector<rt::Tensor> ar_outputs;
  std::vector<rt::OutputSlice> output_slices;
  std::vector<int32_t> generated_tokens;
  std::vector<int32_t> pending_pbd_tokens;
  std::vector<int32_t> pbd_input_tokens;
  std::vector<int32_t> ar_input_tokens;
  rt::Tensor pending_ar_logits;
  uint32_t random_state = 0x9e3779b9u;
  uint64_t dump_invocation = 0;
};

/**
 * @brief Locate the first KV output for host- or BPU-sampled graph layouts.
 * @param outputs Graph outputs in vendor order.
 * @return Index of the first KV tensor.
 */
size_t CacheOutputOffset(const std::vector<rt::Tensor>& outputs) {
  return !outputs.empty() && outputs[0].dtype == kS32 &&
                 SameShape(outputs[0].shape, {1, 6, 1})
             ? 7
             : 1;
}

/**
 * @brief Compute a compact FNV-1a marker for optional graph dumps.
 * @param cache Current Language KV-cache state.
 * @return Deterministic cache fingerprint.
 */
uint64_t FingerprintCache(const CacheState& cache) {
  // A small identity marker for a graph input state.  Full cache dumps are
  // deliberately avoided because a 4096-token cache is large on the board.
  uint64_t value = 1469598103934665603ULL;
  for (const rt::Tensor& tensor : cache.tensors) {
    for (uint8_t byte : tensor.data) {
      value ^= static_cast<uint64_t>(byte);
      value *= 1099511628211ULL;
    }
  }
  return value;
}

/**
 * @brief Write opt-in graph logits and metadata when LA_GRAPH_DUMP_DIR is set.
 * @param graph_name Executed graph name.
 * @param token_base Synthetic-token base used by diagnostic calls.
 * @param past_len Number of committed cache rows before execution.
 * @param pbd Whether this was a PBD graph.
 * @param pbd_prefix_len Accepted prefix carried into the PBD graph.
 * @param explicit_tokens Explicit graph input tokens, when present.
 * @param cache Cache state supplied to the graph.
 * @param outputs Graph outputs in vendor order.
 * @param invocation Monotonic dump sequence counter.
 * @return True when dumping is disabled or every requested file was written.
 */
bool DumpGraphDebug(const std::string& graph_name, int32_t token_base,
                     int32_t past_len, bool pbd, int32_t pbd_prefix_len,
                     const std::vector<int32_t>* explicit_tokens,
                     const CacheState& cache,
                     const std::vector<rt::Tensor>& outputs,
                     uint64_t* invocation) {
  const char* raw_dir = std::getenv("LA_GRAPH_DUMP_DIR");
  if (raw_dir == nullptr || raw_dir[0] == '\0') return true;
  if (outputs.empty()) return false;

  if (invocation == nullptr) return false;
  const uint64_t current = ++*invocation;
  const std::filesystem::path directory(raw_dir);
  std::error_code error;
  std::filesystem::create_directories(directory, error);
  if (error) {
    std::fprintf(stderr, "[FAIL] cannot create LA_GRAPH_DUMP_DIR=%s: %s\n",
                 raw_dir, error.message().c_str());
    return false;
  }

  char stem[128] = {};
  std::snprintf(stem, sizeof(stem), "%04llu_%s",
                static_cast<unsigned long long>(current), graph_name.c_str());
  const std::filesystem::path logits_path = directory / (std::string(stem) + ".logits.f16.bin");
  const std::filesystem::path metadata_path = directory / (std::string(stem) + ".json");
  const rt::Tensor& logits = outputs[0];
  std::ofstream logits_file(logits_path, std::ios::binary | std::ios::trunc);
  if (!logits_file) return false;
  logits_file.write(reinterpret_cast<const char*>(logits.data.data()),
                    static_cast<std::streamsize>(logits.data.size()));
  if (!logits_file) return false;

  std::ofstream metadata(metadata_path, std::ios::trunc);
  if (!metadata) return false;
  metadata << "{\n"
           << "  \"graph\": \"" << graph_name << "\",\n"
           << "  \"invocation\": " << current << ",\n"
           << "  \"token_base\": " << token_base << ",\n"
           << "  \"past_len\": " << past_len << ",\n"
           << "  \"pbd\": " << (pbd ? "true" : "false") << ",\n"
           << "  \"pbd_prefix_len\": " << pbd_prefix_len << ",\n"
           << "  \"cache_fnv1a64\": \"0x" << std::hex
           << static_cast<unsigned long long>(FingerprintCache(cache)) << std::dec << "\",\n"
           << "  \"logits_dtype\": " << logits.dtype << ",\n"
           << "  \"logits_shape\": [";
  for (size_t index = 0; index < logits.shape.size(); ++index) {
    metadata << logits.shape[index]
             << (index + 1 == logits.shape.size() ? "" : ", ");
  }
  metadata << "],\n  \"explicit_tokens\": [";
  if (explicit_tokens != nullptr) {
    for (size_t index = 0; index < explicit_tokens->size(); ++index) {
      metadata << explicit_tokens->at(index)
               << (index + 1 == explicit_tokens->size() ? "" : ", ");
    }
  }
  metadata << "]\n}\n";
  return static_cast<bool>(metadata);
}

struct InputPayload {
  const std::vector<int32_t>& prompt_ids;
  const std::vector<uint8_t>& visual_features;
};

struct GenerationMetrics {
  int32_t prefill_tokens = 0;
  double prefill_ms = 0.0;
  int32_t decode_tokens = 0;
  double decode_ms = 0.0;
  int32_t pbd_calls = 0;
  int32_t pbd_accepted_tokens = 0;
  int32_t ar_calls = 0;
  int32_t ar_tokens = 0;
  double cache_initialize_ms = 0.0;
  double cache_seed_ms = 0.0;
  struct GraphTiming {
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
  std::map<std::string, GraphTiming> graph_timings;
  std::map<std::string, int32_t> decode_events;
  double cache_update_ms = 0.0;
  double host_decode_ms = 0.0;
};

using TokenCallback = std::function<void(int32_t)>;

/**
 * @brief Emit up to count accepted tokens through an optional callback.
 * @param callback Consumer invoked once per accepted token.
 * @param tokens Candidate token sequence.
 * @param count Maximum number of tokens to emit.
 */
void EmitTokens(const TokenCallback& callback,
                const std::vector<int32_t>& tokens,
                size_t count) {
  if (!callback) return;
  for (size_t index = 0; index < std::min(count, tokens.size()); ++index) {
    callback(tokens[index]);
  }
}

/**
 * @brief Recognize complete empty, point, or rectangle box token structures.
 * @param tokens Candidate output tokens.
 * @return True when the sequence is a complete LocateAnything box structure.
 */
bool IsCompleteDetectionBox(const std::vector<int32_t>& tokens) {
  if (tokens.size() == 3 && tokens[0] == kBoxStartToken &&
      tokens[1] == kNoneToken && tokens[2] == kBoxEndToken) {
    return true;
  }
  if (tokens.size() == 4 && tokens[0] == kBoxStartToken &&
      rt::IsCoordinateToken(tokens[1]) && rt::IsCoordinateToken(tokens[2]) &&
      tokens[3] == kBoxEndToken) {
    return true;
  }
  return tokens.size() == 6 && tokens[0] == kBoxStartToken &&
         rt::IsCoordinateToken(tokens[1]) && rt::IsCoordinateToken(tokens[2]) &&
         rt::IsCoordinateToken(tokens[3]) && rt::IsCoordinateToken(tokens[4]) &&
         tokens[5] == kBoxEndToken;
}

/**
 * @brief Check whether a candidate box already appears in the response.
 * @param response Tokens accepted so far.
 * @param candidate Candidate box tokens.
 * @return True when an identical complete box is already present.
 */
bool HasRepeatedDetectionBox(const std::vector<int32_t>& response,
                             const std::vector<int32_t>& candidate) {
  if (!IsCompleteDetectionBox(candidate) || response.size() < candidate.size()) {
    return false;
  }
  return std::search(response.begin(), response.end(), candidate.begin(),
                     candidate.end()) != response.end();
}

/**
 * @brief Check whether the response ends with a previously emitted box.
 * @param response Tokens accepted so far.
 * @return True when the trailing complete box is a duplicate.
 */
bool HasRepeatedTrailingDetectionBox(const std::vector<int32_t>& response) {
  for (const size_t length : {size_t{6}, size_t{4}, size_t{3}}) {
    if (response.size() < length) continue;
    const auto begin = response.end() - static_cast<std::ptrdiff_t>(length);
    const std::vector<int32_t> candidate(begin, response.end());
    if (!IsCompleteDetectionBox(candidate)) continue;
    if (std::search(response.begin(), begin, candidate.begin(), candidate.end()) != begin) {
      return true;
    }
  }
  return false;
}

/**
 * @brief Build deterministic synthetic embeddings for runtime diagnostics.
 * @param graph Graph whose input contract determines the output tensor.
 * @param embed Memory-mapped embedding table.
 * @param token_base First synthetic token ID.
 * @param output Destination embedding tensor.
 * @return True when the graph contract is supported and output was built.
 */
bool BuildEmbeddings(const rt::Graph& graph, const rt::EmbedLookup& embed,
                     int32_t token_base, rt::Tensor* output) {
  const auto& shape = graph.GetInputShapes()[0];
  if (graph.GetInputDtypes()[0] != kF16 || shape.size() != 3 ||
      shape[0] != 1 || shape[2] != kHidden) {
    return false;
  }
  const int32_t query = shape[1];
  std::vector<int32_t> token_ids(static_cast<size_t>(query));
  for (int32_t index = 0; index < query; ++index) {
    token_ids[static_cast<size_t>(index)] = (token_base + index) % kVocab;
  }
  output->shape = shape;
  output->dtype = kF16;
  output->data.resize(static_cast<size_t>(ElementCount(shape)) * 2);
  embed.Gather(token_ids.data(), query, output->data.data());
  return true;
}

/**
 * @brief Gather embeddings for an exact decode token sequence.
 * @param graph Decode graph defining query length and dtype.
 * @param embed Memory-mapped embedding table.
 * @param ids Token IDs whose count must equal graph query length.
 * @param output Destination embedding tensor.
 * @return True when inputs satisfy the graph contract.
 */
bool BuildExplicitEmbeddings(const rt::Graph& graph, const rt::EmbedLookup& embed,
                             const std::vector<int32_t>& ids,
                             rt::Tensor* output) {
  const auto& shape = graph.GetInputShapes()[0];
  if (graph.GetInputDtypes()[0] != kF16 || shape.size() != 3 ||
      shape[0] != 1 || shape[2] != kHidden ||
      ids.size() != static_cast<size_t>(shape[1])) {
    return false;
  }
  output->shape = shape;
  output->dtype = kF16;
  output->data.resize(static_cast<size_t>(ElementCount(shape)) * sizeof(uint16_t));
  embed.Gather(ids.data(), static_cast<int32_t>(ids.size()), output->data.data());
  return true;
}

/**
 * @brief Right-align prompt embeddings and replace image tokens with Vision features.
 * @param graph Prefill graph defining the fixed query length.
 * @param embed Memory-mapped text embedding table.
 * @param payload Prompt IDs and Vision features; null selects diagnostic data.
 * @param output Destination prefill embedding tensor.
 * @param active_len Receives the number of non-padding prompt rows.
 * @return True when prompt and graph dimensions are compatible.
 */
bool BuildPrefillEmbeddings(const rt::Graph& graph, const rt::EmbedLookup& embed,
                             const InputPayload* payload, rt::Tensor* output,
                             int32_t* active_len,
                             bool fused_initial_pbd) {
  const auto& shape = graph.GetInputShapes()[0];
  if (graph.GetInputDtypes()[0] != kF16 || shape.size() != 3 ||
      shape[0] != 1 || shape[1] < 128 || shape[2] != kHidden) {
    return false;
  }
  const int32_t query = shape[1];
  output->shape = shape;
  output->dtype = kF16;
  const size_t row_bytes = static_cast<size_t>(kHidden) * sizeof(uint16_t);
  const size_t total_bytes = static_cast<size_t>(ElementCount(shape)) *
                             sizeof(uint16_t);
  output->data.resize(total_bytes);
  if (payload == nullptr) {
    std::memset(output->data.data(), 0, total_bytes);
    std::vector<int32_t> token_ids(static_cast<size_t>(query));
    for (int32_t index = 0; index < query; ++index) token_ids[index] = index % kVocab;
    embed.Gather(token_ids.data(), query, output->data.data());
    *active_len = query;
    return true;
  }
  const int32_t length = static_cast<int32_t>(payload->prompt_ids.size());
  const int32_t pbd_rows = fused_initial_pbd ? 6 : 0;
  if (length <= 0 || length + pbd_rows > query) return false;
  const int32_t row_offset = shape[1] - length;
  // Every active row is replaced by either text or Vision data. Clear only
  // the left padding rather than the full fixed-size Prefill tensor.
  std::memset(output->data.data(), 0,
              static_cast<size_t>(row_offset) * row_bytes);
  // Image-token embeddings are overwritten below. Gather only non-image runs
  // directly into the persistent graph-input buffer.
  int32_t run_start = 0;
  auto gather_text_run = [&](int32_t run_end) {
    if (run_end <= run_start) return;
    embed.Gather(payload->prompt_ids.data() + run_start, run_end - run_start,
                 output->data.data() +
                     static_cast<size_t>(row_offset + run_start) * row_bytes);
  };
  for (int32_t index = 0; index < length; ++index) {
    if (payload->prompt_ids[static_cast<size_t>(index)] != kImageToken) continue;
    gather_text_run(index);
    run_start = index + 1;
  }
  gather_text_run(length);
  if (fused_initial_pbd) {
    const std::vector<int32_t> pbd_ids{
        payload->prompt_ids.back(), kTextMaskToken, kTextMaskToken,
        kTextMaskToken, kTextMaskToken, kTextMaskToken};
    embed.Gather(
        pbd_ids.data(), static_cast<int32_t>(pbd_ids.size()),
        output->data.data());
  }
  const auto* visual = payload->visual_features.data();
  size_t visual_index = 0;
  auto* destination = reinterpret_cast<uint16_t*>(output->data.data());
  for (int32_t index = 0; index < length; ++index) {
    if (payload->prompt_ids[index] != kImageToken) continue;
    std::memcpy(destination +
                    static_cast<size_t>(row_offset + index) * kHidden,
                visual + visual_index * static_cast<size_t>(kHidden) *
                             sizeof(uint16_t),
                static_cast<size_t>(kHidden) * sizeof(uint16_t));
    ++visual_index;
  }
  *active_len = length;
  return true;
}

/**
 * @brief Build q1 AR or q6 PBD embeddings from the latest prompt token.
 * @param graph Decode graph defining the fixed query length.
 * @param embed Memory-mapped embedding table.
 * @param payload Prompt payload containing the latest token.
 * @param pbd True for the six-row masked PBD input.
 * @param output Destination decode embedding tensor.
 * @return True when the requested mode matches the graph contract.
 */
bool BuildDecodeEmbeddings(const rt::Graph& graph, const rt::EmbedLookup& embed,
                           const InputPayload* payload, bool pbd,
                           rt::Tensor* output) {
  if (payload == nullptr || payload->prompt_ids.empty()) return false;
  const auto& shape = graph.GetInputShapes()[0];
  if (graph.GetInputDtypes()[0] != kF16 || shape.size() != 3 ||
      shape[0] != 1 || shape[2] != kHidden) {
    return false;
  }
  const int32_t query = shape[1];
  if ((pbd && query != 6) || (!pbd && query != 1)) return false;
  int32_t ids[6];
  std::fill(ids, ids + query, kTextMaskToken);
  ids[0] = payload->prompt_ids.back();
  output->shape = shape;
  output->dtype = kF16;
  output->data.resize(static_cast<size_t>(ElementCount(shape)) * sizeof(uint16_t));
  embed.Gather(ids, query, output->data.data());
  return true;
}

/**
 * @brief Build fixed-shape prefill or decode position IDs.
 * @param graph Graph defining position tensor dimensions and dtype.
 * @param start First committed position for decode.
 * @param pbd Whether PBD shared-position semantics apply.
 * @param active_len Non-padding prefill rows, or -1 for decode.
 * @param pbd_prefix_len Accepted prefix included in an extended PBD graph.
 * @param output Destination position tensor.
 * @return True when positions satisfy the graph contract.
 */
bool BuildPositions(const rt::Graph& graph, int32_t start, bool pbd,
                     int32_t active_len, int32_t pbd_prefix_len,
                     bool fused_initial_pbd, rt::Tensor* output) {
  const auto& shape = graph.GetInputShapes()[1];
  if (graph.GetInputDtypes()[1] != kS32 || shape.size() != 3 ||
      shape[0] != 1 || shape[1] != 1) {
    return false;
  }
  output->shape = shape;
  output->dtype = kS32;
  const int32_t query = shape[2];
  output->data.resize(static_cast<size_t>(query) * sizeof(int32_t));
  auto* values = reinterpret_cast<int32_t*>(output->data.data());
  if (query >= 128 && active_len > 0) {
    const int32_t pbd_rows = fused_initial_pbd ? 6 : 0;
    if (active_len + pbd_rows > query || start != 0 ||
        (pbd && !fused_initial_pbd)) return false;
    const int32_t row_offset = query - active_len;
    for (int32_t index = 0; index < query; ++index) {
      if (fused_initial_pbd && index < pbd_rows) {
        values[index] = active_len - 1 + index;
      } else if (index < row_offset) {
        values[index] = 0;
      } else {
        values[index] = index - row_offset;
      }
    }
  } else {
    if (pbd_prefix_len < 0 || pbd_prefix_len > query ||
        (pbd_prefix_len != 0 && !pbd)) {
      return false;
    }
    for (int32_t index = 0; index < query; ++index) {
      values[index] = start + index -
                      (pbd && index >= pbd_prefix_len ? 1 : 0);
    }
  }
  return true;
}

/**
 * @brief Build a right-aligned prefill mask or a PBD-aware decode mask.
 * @param graph Graph defining mask dimensions and dtype.
 * @param past_len Number of committed cache rows.
 * @param block_size PBD block width; zero selects causal behavior.
 * @param active_len Non-padding prefill rows, or -1 for decode.
 * @param output Destination fp16 attention-mask tensor.
 * @return True when the requested mask fits the graph contract.
 */
bool BuildMask(const rt::Graph& graph, int32_t past_len, int32_t block_size,
                int32_t active_len, bool fused_initial_pbd,
                rt::Tensor* output) {
  const auto& shape = graph.GetInputShapes()[2];
  if (graph.GetInputDtypes()[2] != kF16 || shape.size() != 3 || shape[0] != 1) {
    return false;
  }
  const int32_t query = shape[1];
  const int32_t cache_len = shape[2];
  const size_t element_count = static_cast<size_t>(query) * cache_len;
  output->shape = shape;
  output->dtype = kF16;
  output->data.resize(element_count * sizeof(uint16_t));
  auto* mask = reinterpret_cast<uint16_t*>(output->data.data());
  if (query >= 128 && active_len >= 0) {
    const int32_t pbd_rows = fused_initial_pbd ? 6 : 0;
    if (active_len + pbd_rows > query || past_len != 0) return false;
    std::fill(mask, mask + element_count, kMaskValue);
    const int32_t current_start = cache_len - query;
    const int32_t row_offset = query - active_len;
    for (int32_t row_index = 0; row_index < query; ++row_index) {
      uint16_t* row = mask + static_cast<size_t>(row_index) * cache_len;
      if (fused_initial_pbd && row_index < pbd_rows) {
        for (int32_t index = 0; index < pbd_rows; ++index) {
          row[current_start + index] = 0;
        }
        for (int32_t index = row_offset; index < query - 1; ++index) {
          row[current_start + index] = 0;
        }
        continue;
      }
      if (row_index < row_offset) {
        row[current_start + row_index] = 0;
        continue;
      }
      for (int32_t index = row_offset; index <= row_index; ++index) {
        row[current_start + index] = 0;
      }
    }
  } else if (!rt::BuildAttentionMaskData(
                 query, cache_len, past_len, block_size, kMaskValue, false,
                 mask, element_count)) {
    return false;
  }
  return true;
}

/**
 * @brief Allocate zeroed device-resident KV inputs for prefill.
 * @param graph Prefill graph defining all 72 cache tensors.
 * @param state Destination cache state.
 * @return True when every cache buffer was allocated.
 */
bool BuildZeroCaches(const rt::Graph& graph, CacheState* state) {
  const auto& shapes = graph.GetInputShapes();
  const auto& dtypes = graph.GetInputDtypes();
  if (shapes.size() != 3 + kCacheCount) return false;

  bool reusable = state->tensors.size() == kCacheCount;
  if (reusable) {
    for (int32_t index = 0; index < kCacheCount; ++index) {
      const size_t input_index = static_cast<size_t>(index + 3);
      const int32_t element_bytes = rt::DtypeElementBytes(dtypes[input_index]);
      const size_t cache_bytes = element_bytes > 0
          ? static_cast<size_t>(ElementCount(shapes[input_index])) *
                static_cast<size_t>(element_bytes)
          : 0;
      const rt::Tensor& tensor = state->tensors[static_cast<size_t>(index)];
      if (element_bytes <= 0 || tensor.shape != shapes[input_index] ||
          tensor.dtype != dtypes[input_index] || tensor.device_buffer == nullptr ||
          tensor.device_buffer->size() != cache_bytes) {
        reusable = false;
        break;
      }
    }
  }
  if (reusable) {
    for (rt::Tensor& tensor : state->tensors) {
      // Graph inputs are read-only and use separate output buffers. The
      // allocation was zeroed and cache-cleaned once, so only its logical view
      // needs resetting between independent Prefill calls.
      tensor.byte_offset = 0;
    }
    return true;
  }

  state->tensors.clear();
  state->tensors.reserve(kCacheCount);
  for (size_t index = 3; index < shapes.size(); ++index) {
    const int32_t element_bytes = rt::DtypeElementBytes(dtypes[index]);
    if (element_bytes <= 0 || shapes[index].empty()) return false;
    rt::Tensor tensor;
    tensor.shape = shapes[index];
    tensor.dtype = dtypes[index];
    const size_t cache_bytes = static_cast<size_t>(ElementCount(tensor.shape)) *
                               static_cast<size_t>(element_bytes);
    if (!rt::AllocateDeviceBuffer(cache_bytes, true,
                                  &tensor.device_buffer).ok()) {
      return false;
    }
    state->tensors.push_back(std::move(tensor));
  }
  return true;
}

/**
 * @brief Seed mirrored device KV caches from valid prefill output rows.
 * @param graph Prefill graph defining the full cache contract.
 * @param updates Prefill logits followed by 72 KV tensors.
 * @param cache_offset Number of valid right-aligned prefill rows.
 * @param state Destination mirrored cache state.
 * @return True when all KV tensors were validated and copied.
 */
bool BuildFullCaches(const rt::Graph& graph,
                     const std::vector<rt::Tensor>& updates,
                     int32_t cache_offset,
                     CacheState* state) {
  const auto& input_shapes = graph.GetInputShapes();
  const auto& input_dtypes = graph.GetInputDtypes();
  if (input_shapes.size() != 3 + kCacheCount ||
      updates.size() != 1 + kCacheCount) {
    return false;
  }
  bool reusable = state->tensors.size() == kCacheCount;
  if (reusable) {
    for (int32_t index = 0; index < kCacheCount; ++index) {
      const size_t input_index = static_cast<size_t>(index + 3);
      const int32_t element_bytes = rt::DtypeElementBytes(input_dtypes[input_index]);
      const size_t cache_bytes = element_bytes > 0
          ? static_cast<size_t>(ElementCount(input_shapes[input_index])) *
                static_cast<size_t>(element_bytes)
          : 0;
      const rt::Tensor& cache = state->tensors[static_cast<size_t>(index)];
      if (element_bytes <= 0 || cache.shape != input_shapes[input_index] ||
          cache.dtype != input_dtypes[input_index] ||
          cache.device_buffer == nullptr ||
          cache.device_buffer->size() != cache_bytes * 2) {
        reusable = false;
        break;
      }
    }
  }
  if (!reusable) {
    state->tensors.clear();
    state->tensors.reserve(kCacheCount);
  }
  for (int32_t index = 0; index < kCacheCount; ++index) {
    const size_t output_index = static_cast<size_t>(index + 1);
    const size_t input_index = static_cast<size_t>(index + 3);
    const int32_t element_bytes = rt::DtypeElementBytes(input_dtypes[input_index]);
    if (element_bytes <= 0 || updates[output_index].dtype != input_dtypes[input_index] ||
        updates[output_index].shape.size() != 4 ||
        input_shapes[input_index].size() != 4) {
      return false;
    }
    const int32_t query = updates[output_index].shape[1];
    const int32_t valid_len = cache_offset;
    const int32_t cache_len = input_shapes[input_index][1];
    if (query <= 0 || valid_len <= 0 || valid_len > query || valid_len > cache_len ||
        ElementCount(updates[output_index].shape) * element_bytes !=
            static_cast<int64_t>(updates[output_index].data.size())) {
      return false;
    }
    rt::Tensor cache;
    if (reusable) {
      cache = std::move(state->tensors[static_cast<size_t>(index)]);
    } else {
      cache.shape = input_shapes[input_index];
      cache.dtype = input_dtypes[input_index];
    }
    const size_t cache_bytes = static_cast<size_t>(ElementCount(cache.shape)) *
                               static_cast<size_t>(element_bytes);
    if (!reusable &&
        !rt::AllocateDeviceBuffer(cache_bytes * 2, true,
                                  &cache.device_buffer).ok()) {
      return false;
    }
    cache.byte_offset = 0;
    const size_t row_bytes = static_cast<size_t>(ElementCount(updates[output_index].shape)) /
                             static_cast<size_t>(query) *
                             static_cast<size_t>(element_bytes);
    const size_t destination = static_cast<size_t>(cache_len - valid_len) * row_bytes;
    const size_t copy_bytes = static_cast<size_t>(valid_len) * row_bytes;
    if (destination + copy_bytes > cache_bytes) {
      return false;
    }
    if (!rt::WriteDeviceBuffer(cache.device_buffer, destination,
                               updates[output_index].data.data(), copy_bytes).ok() ||
        !rt::WriteDeviceBuffer(cache.device_buffer, cache_bytes + destination,
                               updates[output_index].data.data(), copy_bytes).ok()) {
      return false;
    }
    if (reusable) {
      state->tensors[static_cast<size_t>(index)] = std::move(cache);
    } else {
      state->tensors.push_back(std::move(cache));
    }
  }
  return true;
}

/**
 * @brief Assemble ordered embedding, position, mask, cache, and sampling inputs.
 * @param graph Target Language graph.
 * @param embed Memory-mapped embedding table.
 * @param token_base Synthetic token base used only by diagnostic calls.
 * @param past_len Number of committed cache rows.
 * @param pbd Whether the target graph performs PBD.
 * @param cache Current KV-cache tensors.
 * @param payload Prompt and Vision payload, when running real inference.
 * @param explicit_tokens Exact decode tokens, when already selected.
 * @param active_len In/out count of active prefill rows.
 * @param inputs Storage for tensors and ordered non-owning views.
 * @param pbd_prefix_len Accepted prefix included by an extended PBD graph.
 * @param generated_tokens History used by BPU sampling inputs.
 * @param random_state Mutable PRNG state for BPU sampling.
 * @return True when every tensor matches the target graph contract.
 */
bool BuildInputs(const rt::Graph& graph, const rt::EmbedLookup& embed,
                  int32_t token_base, int32_t past_len, bool pbd,
                  const CacheState& cache, const InputPayload* payload,
                  const std::vector<int32_t>* explicit_tokens,
                  int32_t* active_len, PreparedInputs* inputs,
                  int32_t pbd_prefix_len,
                  const std::vector<int32_t>* generated_tokens,
                  uint32_t* random_state,
                  bool fused_initial_pbd) {
  if (cache.tensors.size() != kCacheCount) return false;
  const int32_t query = graph.GetInputShapes()[0][1];
  const bool embedding_ok =
      (explicit_tokens != nullptr
           ? BuildExplicitEmbeddings(graph, embed, *explicit_tokens,
                                     &inputs->embeddings)
           : payload != nullptr && query >= 128
           ? BuildPrefillEmbeddings(graph, embed, payload, &inputs->embeddings,
                                     active_len, fused_initial_pbd)
           : payload != nullptr
                 ? BuildDecodeEmbeddings(graph, embed, payload, pbd,
                                         &inputs->embeddings)
                 : BuildEmbeddings(graph, embed, token_base,
                                   &inputs->embeddings));
  if (!embedding_ok ||
      !BuildPositions(graph, past_len, pbd,
                      payload != nullptr && graph.GetInputShapes()[0][1] >= 128
                          ? *active_len
                          : -1,
                      pbd_prefix_len,
                      fused_initial_pbd,
                      &inputs->positions) ||
      !BuildMask(graph, past_len, pbd ? 6 : 0,
                  payload != nullptr && graph.GetInputShapes()[0][1] >= 128
                      ? *active_len
                      : -1,
                  fused_initial_pbd,
                  &inputs->mask)) {
    return false;
  }
  inputs->views.clear();
  inputs->views.reserve(3 + cache.tensors.size());
  inputs->views.push_back(&inputs->embeddings);
  inputs->views.push_back(&inputs->positions);
  inputs->views.push_back(&inputs->mask);
  for (const auto& tensor : cache.tensors) inputs->views.push_back(&tensor);
  if (graph.GetInputShapes().size() == 3 + kCacheCount + 2) {
    if (generated_tokens == nullptr || graph.GetInputDtypes().size() !=
            graph.GetInputShapes().size()) return false;
    const size_t history_index = 3 + kCacheCount;
    const size_t random_index = history_index + 1;
    inputs->history_mask.shape = graph.GetInputShapes()[history_index];
    inputs->history_mask.dtype = graph.GetInputDtypes()[history_index];
    inputs->history_mask.data.resize(
        static_cast<size_t>(ElementCount(inputs->history_mask.shape)));
    std::memset(inputs->history_mask.data.data(), 0,
                inputs->history_mask.data.size());
    for (int32_t token : *generated_tokens) {
      if (token < 0 || token >= kVocab) continue;
      for (int32_t row = 0; row < 6; ++row) {
        inputs->history_mask.data[static_cast<size_t>(row) * kVocab + token] = 1;
      }
    }
    inputs->random_values.shape = graph.GetInputShapes()[random_index];
    inputs->random_values.dtype = graph.GetInputDtypes()[random_index];
    inputs->random_values.data.resize(
        static_cast<size_t>(ElementCount(inputs->random_values.shape)) * 2);
    std::memset(inputs->random_values.data.data(), 0,
                inputs->random_values.data.size());
    auto* values = reinterpret_cast<uint16_t*>(inputs->random_values.data.data());
    if (random_state == nullptr) return false;
    for (int32_t row = 0; row < 6; ++row) {
      *random_state ^= *random_state << 13;
      *random_state ^= *random_state >> 17;
      *random_state ^= *random_state << 5;
      const float uniform =
          static_cast<float>(*random_state & 0x3ffu) / 1024.0f;
      values[row] = FloatToFp16(uniform);
    }
    inputs->views.push_back(&inputs->history_mask);
    inputs->views.push_back(&inputs->random_values);
  }
  return true;
}

/**
 * @brief Accumulate one graph execution into generation diagnostics.
 * @param name Executed graph name.
 * @param execution_metrics HBM wrapper timings and byte counters.
 * @param input_build_ms Host time spent constructing graph inputs.
 * @param metrics Optional generation metrics destination.
 */
void RecordGraphTiming(const std::string& name,
                       const rt::ExecutionMetrics& execution_metrics,
                       double input_build_ms,
                       GenerationMetrics* metrics) {
  if (metrics == nullptr) return;
  GenerationMetrics::GraphTiming& timing = metrics->graph_timings[name];
  ++timing.calls;
  timing.total_ms += execution_metrics.total_ms;
  timing.input_build_ms += input_build_ms;
  timing.buffer_prepare_ms += execution_metrics.buffer_prepare_ms;
  timing.input_pack_ms += execution_metrics.input_pack_ms;
  timing.input_flush_ms += execution_metrics.input_flush_ms;
  timing.bpu_wait_ms += execution_metrics.bpu_wait_ms;
  timing.submit_ms += execution_metrics.submit_ms;
  timing.output_flush_ms += execution_metrics.output_flush_ms;
  timing.output_unpack_ms += execution_metrics.output_unpack_ms;
  timing.input_bytes += execution_metrics.input_bytes;
  timing.resident_input_bytes += execution_metrics.resident_input_bytes;
  timing.output_bytes += execution_metrics.output_bytes;
}

void RecordDecodeEvent(const std::string& event, GenerationMetrics* metrics) {
  if (metrics != nullptr) ++metrics->decode_events[event];
}

/**
 * @brief Build inputs, execute one named graph, and collect selected outputs.
 * @param engine Loaded Language HBM and embedding state.
 * @param name Target graph name.
 * @param token_base Synthetic token base used by diagnostic calls.
 * @param past_len Number of committed cache rows.
 * @param pbd Whether the graph is a PBD graph.
 * @param cache Current KV-cache state.
 * @param outputs Destination graph outputs.
 * @param payload Optional real prompt and Vision payload.
 * @param active_len Optional destination for active prefill length.
 * @param explicit_tokens Optional exact decode tokens.
 * @param pbd_prefix_len Accepted prefix carried by an extended PBD graph.
 * @param metrics Optional generation diagnostics.
 * @param generated_tokens Optional history for BPU sampling inputs.
 * @return True when input construction, execution, and optional dump succeed.
 */
bool RunGraph(EngineState* engine, const std::string& name, int32_t token_base,
              int32_t past_len, bool pbd, const CacheState& cache,
              std::vector<rt::Tensor>* outputs,
              const InputPayload* payload = nullptr,
              int32_t* active_len = nullptr,
              const std::vector<int32_t>* explicit_tokens = nullptr,
              int32_t pbd_prefix_len = 0,
              GenerationMetrics* metrics = nullptr,
              const std::vector<int32_t>* generated_tokens = nullptr) {
  if (engine == nullptr) return false;
  rt::Graph* graph = engine->session.GetGraph(name);
  if (!graph) {
    std::fprintf(stderr, "[FAIL] graph not found: %s\n", name.c_str());
    return false;
  }
  PreparedInputs& inputs = engine->prepared_inputs;
  const bool fused_initial_pbd =
      name == "prefill" && payload != nullptr &&
      !graph->GetOutputShapes().empty() &&
      graph->GetOutputShapes()[0].size() == 3 &&
      graph->GetOutputShapes()[0][1] == 7 &&
      payload->prompt_ids.size() + 6 <=
          static_cast<size_t>(graph->GetInputShapes()[0][1]);
  const auto input_build_started = std::chrono::steady_clock::now();
  int32_t local_active_len =
      graph->GetInputShapes()[0].size() > 1 && graph->GetInputShapes()[0][1] >= 128
          ? graph->GetInputShapes()[0][1]
          : -1;
  if (!BuildInputs(*graph, engine->embed, token_base, past_len, pbd, cache, payload,
                   explicit_tokens, &local_active_len, &inputs,
                    pbd_prefix_len, generated_tokens,
                    &engine->random_state, fused_initial_pbd)) {
    std::fprintf(stderr, "[FAIL] cannot build %s inputs\n", name.c_str());
    return false;
  }
  const double input_build_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - input_build_started).count();
  if (active_len != nullptr) *active_len = local_active_len;
  rt::ExecutionMetrics execution_metrics;
  std::vector<rt::OutputSlice>& output_slices = engine->output_slices;
  output_slices.clear();
  const std::vector<rt::OutputSlice>* selected_outputs = nullptr;
  if (name == "prefill" && payload != nullptr) {
    const auto& output_shapes = graph->GetOutputShapes();
    output_slices.resize(output_shapes.size());
    for (size_t index = 0; index < output_shapes.size(); ++index) {
      if (output_shapes[index].size() < 2 || local_active_len <= 0) {
        std::fprintf(stderr, "[FAIL] cannot slice prefill output idx=%zu\n",
                     index);
        return false;
      }
      const int32_t output_rows = output_shapes[index][1];
      if (index == 0 && output_rows == 1) {
        output_slices[index] = rt::OutputSlice{0, 1};
      } else if (index == 0 && output_rows == 7) {
        output_slices[index] = rt::OutputSlice{0, 7};
      } else if (local_active_len <= output_rows) {
        const int32_t offset = output_rows - local_active_len;
        if (offset < 0) {
          std::fprintf(stderr, "[FAIL] invalid fused prefill slice idx=%zu\n",
                       index);
          return false;
        }
        output_slices[index] = rt::OutputSlice{offset, local_active_len};
      } else {
        std::fprintf(stderr, "[FAIL] cannot slice prefill output idx=%zu\n",
                     index);
        return false;
      }
    }
    selected_outputs = &output_slices;
  } else if (pbd && pbd_prefix_len == 0) {
    // q6 logits decide which tokens are accepted; its provisional KV rows are
    // never committed. Avoid invalidating and unpacking those rows on Host.
    const auto& output_shapes = graph->GetOutputShapes();
    output_slices.resize(output_shapes.size());
    const size_t cache_offset =
        !output_shapes.empty() && graph->GetOutputDtypes()[0] == kS32 &&
                SameShape(output_shapes[0], {1, 6, 1})
            ? 7
            : 1;
    for (size_t index = cache_offset; index < output_shapes.size(); ++index) {
      output_slices[index] = rt::OutputSlice{0, -1, false};
    }
    selected_outputs = &output_slices;
  } else if (pbd && pbd_prefix_len > 0) {
    // Extended PBD graphs causally commit the accepted prefix, then evaluate
    // a six-row decision window. Materialize only those two used regions.
    const auto& output_shapes = graph->GetOutputShapes();
    const auto& output_dtypes = graph->GetOutputDtypes();
    output_slices.resize(output_shapes.size());
    bool sliced = false;
    if (output_dtypes.size() == output_shapes.size() &&
        !output_shapes.empty() && output_dtypes[0] == kF16 &&
        output_shapes[0].size() >= 2 &&
        output_shapes[0][1] >= pbd_prefix_len + 6) {
      output_slices[0] = rt::OutputSlice{pbd_prefix_len, 6};
      sliced = true;
    }
    const size_t cache_offset =
        output_dtypes.size() == output_shapes.size() &&
                !output_shapes.empty() && output_dtypes[0] == kS32 &&
                SameShape(output_shapes[0], {1, 6, 1})
            ? 7
            : 1;
    for (size_t index = cache_offset; index < output_shapes.size(); ++index) {
      if (output_shapes[index].size() >= 2 &&
          output_shapes[index][1] > pbd_prefix_len) {
        output_slices[index] = rt::OutputSlice{0, pbd_prefix_len};
        sliced = true;
      }
    }
    if (sliced) selected_outputs = &output_slices;
  } else if (!pbd && explicit_tokens != nullptr &&
             explicit_tokens->size() > 1) {
    // A bridge AR graph commits every supplied token, but generation only
    // consumes the final logits row.
    const auto& output_shapes = graph->GetOutputShapes();
    const auto& output_dtypes = graph->GetOutputDtypes();
    const int32_t query = static_cast<int32_t>(explicit_tokens->size());
    if (output_dtypes.size() == output_shapes.size() &&
        !output_shapes.empty() && output_dtypes[0] == kF16 &&
        output_shapes[0].size() >= 2 && output_shapes[0][1] >= query) {
      output_slices.resize(output_shapes.size());
      output_slices[0] = rt::OutputSlice{query - 1, 1};
      selected_outputs = &output_slices;
    }
  }
  const rt::Result result = engine->session.ExecuteGraphByName(
      name, inputs.views, outputs, &execution_metrics, selected_outputs);
  if (!result.ok()) {
    std::fprintf(stderr, "[FAIL] %s execute code=%d: %s\n", name.c_str(),
                 result.code, result.message.c_str());
    return false;
  }
  RecordGraphTiming(name, execution_metrics, input_build_ms, metrics);
  if (!DumpGraphDebug(name, token_base, past_len, pbd, pbd_prefix_len,
                      explicit_tokens, cache, *outputs,
                      &engine->dump_invocation)) {
    std::fprintf(stderr, "[FAIL] cannot dump debug outputs for %s\n",
                 name.c_str());
    return false;
  }
  if (std::getenv("LA_PROFILE_EXECUTION") != nullptr) {
    std::printf(
        "[profile] graph=%s total=%.3f input_build=%.3f prepare=%.3f pack=%.3f "
        "input_flush=%.3f submit=%.3f bpu_wait=%.3f output_flush=%.3f "
        "unpack=%.3f input_mib=%.2f resident_input_mib=%.2f output_mib=%.2f\n",
        name.c_str(), execution_metrics.total_ms, input_build_ms,
        execution_metrics.buffer_prepare_ms, execution_metrics.input_pack_ms,
        execution_metrics.input_flush_ms, execution_metrics.submit_ms,
        execution_metrics.bpu_wait_ms, execution_metrics.output_flush_ms,
        execution_metrics.output_unpack_ms,
        execution_metrics.input_bytes / (1024.0 * 1024.0),
        execution_metrics.resident_input_bytes / (1024.0 * 1024.0),
        execution_metrics.output_bytes / (1024.0 * 1024.0));
  }
  return true;
}

/**
 * @brief Commit accepted graph KV rows into all mirrored cache tensors.
 * @param outputs Graph outputs containing logits and KV updates.
 * @param history_len Number of committed rows before this update.
 * @param state Destination mirrored cache state.
 * @param valid_query Accepted prefix rows; negative commits all query rows.
 * @param metrics Optional cache-copy timing destination.
 * @return True when every KV tensor accepted the update.
 */
bool AppendCacheUpdate(const std::vector<rt::Tensor>& outputs,
                       int32_t history_len, CacheState* state,
                       int32_t valid_query = -1,
                       GenerationMetrics* metrics = nullptr) {
  const auto started = std::chrono::steady_clock::now();
  uint64_t copied_bytes = 0;
  const size_t output_offset = CacheOutputOffset(outputs);
  if (outputs.size() != output_offset + kCacheCount ||
      state->tensors.size() != kCacheCount) {
    return false;
  }
  for (int32_t index = 0; index < kCacheCount; ++index) {
    const rt::Tensor& update = outputs[output_offset + static_cast<size_t>(index)];
    rt::Tensor& cache = state->tensors[static_cast<size_t>(index)];
    if (update.dtype != cache.dtype || update.shape.size() != 4 ||
        cache.shape.size() != 4) {
      return false;
    }
    const int32_t query = update.shape[1];
    const int32_t committed_query = valid_query < 0 ? query : valid_query;
    const int32_t cache_len = cache.shape[1];
    const int32_t bytes = rt::DtypeElementBytes(cache.dtype);
    if (query <= 0 || committed_query <= 0 || committed_query > query ||
        bytes <= 0 || history_len < 0 ||
        history_len + committed_query > cache_len) {
      return false;
    }
    const size_t row_bytes = static_cast<size_t>(ElementCount(cache.shape)) /
                             static_cast<size_t>(cache_len) *
                             static_cast<size_t>(bytes);
    const bool appended = cache.device_buffer != nullptr
        ? rt::AppendMirroredDeviceRingRows(
              &cache, static_cast<size_t>(cache_len), row_bytes,
              update.data.data(), update.data.size(),
              static_cast<size_t>(committed_query), &copied_bytes)
        : rt::AppendMirroredRingRows(
              &cache.data, static_cast<size_t>(cache_len), row_bytes,
              update.data.data(), update.data.size(),
              static_cast<size_t>(committed_query), &cache.byte_offset,
              &copied_bytes);
    if (!appended) {
      return false;
    }
  }
  const double elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  if (metrics != nullptr) metrics->cache_update_ms += elapsed_ms;
  if (std::getenv("LA_PROFILE_EXECUTION") != nullptr) {
    std::printf("[profile] cache_commit total=%.3f copied_mib=%.3f\n",
                elapsed_ms, copied_bytes / (1024.0 * 1024.0));
  }
  return true;
}

/**
 * @brief Materialize one vocabulary row from a multi-row logits tensor.
 * @param logits Source fp16 logits tensor.
 * @param row Sequence-axis row to copy.
 * @param selected Destination shaped as [1, 1, vocab].
 * @return True when the row and tensor contract are valid.
 */
bool SelectLogitsRow(const rt::Tensor& logits, int32_t row,
                     rt::Tensor* selected) {
  if (logits.dtype != kF16 || logits.shape.size() != 3 ||
      logits.shape[0] != 1 || logits.shape[2] != kVocab ||
      row < 0 || row >= logits.shape[1] ||
      logits.data.size() != static_cast<size_t>(ElementCount(logits.shape)) * 2) {
    return false;
  }
  selected->shape = {1, 1, kVocab};
  selected->dtype = kF16;
  const size_t row_bytes = static_cast<size_t>(kVocab) * sizeof(uint16_t);
  const auto begin = logits.data.begin() + static_cast<size_t>(row) * row_bytes;
  selected->data.assign(begin, begin + row_bytes);
  return true;
}

/**
 * @brief Select the first decision row in q6 or extended PBD logits.
 * @param logits PBD logits tensor.
 * @param prefix_len Prefix rows already accepted.
 * @return First row used by the six-row Host decoder.
 */
int32_t PbdLogitStart(const rt::Tensor& logits, int32_t prefix_len) {
  if (logits.shape.size() != 3 || logits.shape[1] != 6) return prefix_len;
  return 0;
}

/**
 * @brief Select the next-token row from q1 or multi-token AR output.
 * @param logits AR logits tensor.
 * @param accepted Number of tokens committed by the bridge graph.
 * @return Row containing logits for the next token.
 */
int32_t ArLogitRow(const rt::Tensor& logits, int32_t accepted) {
  if (logits.shape.size() != 3 || logits.shape[1] != 1) {
    return accepted - 1;
  }
  return 0;
}

/** Return the sequence capacity declared by the first KV-cache tensor. */
int32_t CacheCapacity(const CacheState& cache);

/**
 * @brief Resolve the fixed PBD graph name for an accepted prefix length.
 * @param prefix_len Number of accepted prefix tokens, from zero through six.
 * @return HBM graph name.
 */
std::string PbdGraphName(int32_t prefix_len) {
  return prefix_len == 0 ? "decode"
                         : "decode_pbd_q" + std::to_string(6 + prefix_len);
}

/**
 * @brief Resolve the fixed AR graph name for a query length.
 * @param q_len Number of AR tokens evaluated together.
 * @return HBM graph name.
 */
std::string ArGraphName(int32_t q_len) {
  return q_len == 1 ? "decode_ar" : "decode_ar_q" + std::to_string(q_len);
}

int32_t CacheCapacity(const CacheState& cache) {
  if (cache.tensors.empty() || cache.tensors.front().shape.size() < 2) return 0;
  return cache.tensors.front().shape[1];
}

/**
 * @brief Generate with PBD and temporary AR fallback for incomplete boxes.
 * @param engine Loaded Language runtime state.
 * @param payload Prompt and Vision features.
 * @param max_new_tokens Hard output-token limit.
 * @param cache Mutable committed KV-cache state.
 * @param history_len Mutable committed-token count.
 * @param response Destination generated tokens.
 * @param stop_reason Destination terminal reason.
 * @param protect_detection_structure Enable duplicate/incomplete-box guards.
 * @param metrics Optional generation diagnostics.
 * @param token_callback Optional accepted-token callback.
 * @return True when generation reached a controlled terminal condition.
 */
bool RunHybridGeneration(EngineState* engine,
                         const InputPayload& payload,
                         int32_t max_new_tokens, CacheState* cache,
                          int32_t* history_len,
                          std::vector<int32_t>* response,
                          std::string* stop_reason,
                          bool protect_detection_structure,
                          GenerationMetrics* metrics,
                          const rt::Tensor* initial_pbd_logits = nullptr,
                          const TokenCallback& token_callback = {}) {
  std::vector<int32_t>& generated = engine->generated_tokens;
  generated.assign(payload.prompt_ids.begin(), payload.prompt_ids.end());
  generated.reserve(payload.prompt_ids.size() +
                    static_cast<size_t>(max_new_tokens));
  std::vector<int32_t>& pending_pbd = engine->pending_pbd_tokens;
  pending_pbd.clear();
  pending_pbd.reserve(6);
  std::vector<int32_t>& pbd_input = engine->pbd_input_tokens;
  pbd_input.clear();
  pbd_input.reserve(12);
  std::vector<int32_t>& ar_input = engine->ar_input_tokens;
  ar_input.clear();
  ar_input.reserve(1);
  const rt::Tensor* pending_ar = nullptr;
  bool use_pbd = true;
  const int32_t cache_len = CacheCapacity(*cache);
  if (cache_len <= 0) return false;
  const rt::Tensor* bootstrap_logits = initial_pbd_logits;

  while (static_cast<int32_t>(response->size()) < max_new_tokens) {
    if (*history_len >= cache_len) {
      *stop_reason = "cache_limit";
      break;
    }
    if (use_pbd) {
      const int32_t prefix_len = static_cast<int32_t>(pending_pbd.size());
      if (prefix_len < 0 || prefix_len > 6 ||
          *history_len + prefix_len + 6 > cache_len) {
        *stop_reason = "cache_limit_before_pbd";
        break;
      }
      pbd_input.clear();
      if (pending_pbd.empty()) {
        pbd_input = {generated.back(), kTextMaskToken, kTextMaskToken,
                     kTextMaskToken, kTextMaskToken, kTextMaskToken};
      } else {
        pbd_input.insert(pbd_input.end(), pending_pbd.begin(), pending_pbd.end());
        pbd_input.push_back(pending_pbd.back());
        pbd_input.insert(pbd_input.end(), 5, kTextMaskToken);
      }
      std::vector<rt::Tensor>& outputs = engine->pbd_outputs;
      bool executed_pbd_graph = false;
      const rt::Tensor* logits = nullptr;
      int32_t pbd_logit_start = 0;
      if (prefix_len == 0 && bootstrap_logits != nullptr) {
        logits = bootstrap_logits;
        bootstrap_logits = nullptr;
        pbd_logit_start = 1;
      } else {
        executed_pbd_graph = true;
        if (!RunGraph(engine, PbdGraphName(prefix_len), 0, *history_len,
                       true, *cache, &outputs, nullptr, nullptr, &pbd_input,
                       prefix_len, metrics, &generated)) {
          return false;
        }
        logits = &outputs[0];
        pbd_logit_start = PbdLogitStart(*logits, prefix_len);
        if (prefix_len > 0) {
          if (!AppendCacheUpdate(outputs, *history_len, cache, prefix_len, metrics)) {
            return false;
          }
          *history_len += prefix_len;
        }
      }
      if (metrics != nullptr) ++metrics->pbd_calls;
      const auto pbd_decode_started = std::chrono::steady_clock::now();
      const rt::HybridDecision decision =
          executed_pbd_graph && !outputs.empty() &&
                  CacheOutputOffset(outputs) == 7
              ? rt::DecodePbdCompact(outputs)
              : rt::DecodePbd(*logits, generated, rt::PbdDecodeConfig{},
                              nullptr, pbd_logit_start);
      RecordDecodeEvent("pbd_" + decision.type, metrics);
      if (metrics != nullptr) {
        metrics->host_decode_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pbd_decode_started).count();
      }
      pending_pbd.clear();

      if (decision.terminal) {
        if (metrics != nullptr) {
          metrics->pbd_accepted_tokens +=
              static_cast<int32_t>(decision.tokens.size());
        }
        response->insert(response->end(), decision.tokens.begin(), decision.tokens.end());
        generated.insert(generated.end(), decision.tokens.begin(), decision.tokens.end());
        EmitTokens(token_callback, decision.tokens, decision.tokens.size());
        *stop_reason = "im_end";
        break;
      }
      if (protect_detection_structure &&
          HasRepeatedDetectionBox(*response, decision.tokens)) {
        *stop_reason = "repeated_box";
        break;
      }
      const int32_t accepted = static_cast<int32_t>(decision.tokens.size());
      const int32_t remaining = max_new_tokens - static_cast<int32_t>(response->size());
      if (accepted <= 0 || accepted > 6) return false;
      if (accepted > remaining) {
        if (metrics != nullptr) metrics->pbd_accepted_tokens += remaining;
        response->insert(response->end(), decision.tokens.begin(),
                         decision.tokens.begin() + remaining);
        EmitTokens(token_callback, decision.tokens, static_cast<size_t>(remaining));
        *stop_reason = "max_new_tokens";
        return true;
      }
      if (decision.switch_to_ar) {
        RecordDecodeEvent("pbd_to_ar", metrics);
        std::vector<rt::Tensor>& bridge = engine->ar_outputs;
        if (!RunGraph(engine, ArGraphName(accepted), 0, *history_len,
                       false, *cache, &bridge, nullptr, nullptr,
                      &decision.tokens, 0, metrics) ||
            !AppendCacheUpdate(bridge, *history_len, cache, accepted, metrics)) {
          return false;
        }
        if (metrics != nullptr) ++metrics->ar_calls;
        *history_len += accepted;
        if (bridge[0].shape == std::vector<int32_t>{1, 1, kVocab}) {
          pending_ar = &bridge[0];
        } else {
          if (!SelectLogitsRow(bridge[0], ArLogitRow(bridge[0], accepted),
                                &engine->pending_ar_logits)) {
            return false;
          }
          pending_ar = &engine->pending_ar_logits;
        }
        use_pbd = false;
      } else {
        pending_pbd = decision.tokens;
      }
      response->insert(response->end(), decision.tokens.begin(), decision.tokens.end());
      generated.insert(generated.end(), decision.tokens.begin(), decision.tokens.end());
      if (metrics != nullptr) metrics->pbd_accepted_tokens += accepted;
      EmitTokens(token_callback, decision.tokens, decision.tokens.size());
      continue;
    }

    if (pending_ar == nullptr) return false;
    const int32_t token = rt::DecodeArGreedy(*pending_ar, generated);
    if (token == kBoxEndToken) {
      RecordDecodeEvent("ar_box_end", metrics);
    } else if (rt::IsCoordinateToken(token)) {
      RecordDecodeEvent("ar_coordinate", metrics);
    } else if (token == kNoneToken) {
      RecordDecodeEvent("ar_none", metrics);
    } else {
      RecordDecodeEvent("ar_other", metrics);
    }
    response->push_back(token);
    generated.push_back(token);
    if (metrics != nullptr) ++metrics->ar_tokens;
    if (protect_detection_structure && token == kBoxEndToken &&
        HasRepeatedTrailingDetectionBox(*response)) {
      *stop_reason = "repeated_box";
      break;
    }
    if (token_callback) token_callback(token);
    if (token != kBoxEndToken && !rt::IsCoordinateToken(token) &&
        token != kNoneToken) {
      *stop_reason = "ar_non_coordinate";
      break;
    }
    if (token == kBoxEndToken) {
      // The next q7 PBD profile causally commits this token and evaluates its
      // following PBD window in one BPU execution.
      pending_pbd = {token};
      RecordDecodeEvent("ar_to_pbd", metrics);
      pending_ar = nullptr;
      use_pbd = true;
      continue;
    }
    ar_input.clear();
    ar_input.push_back(token);
    std::vector<rt::Tensor>& outputs = engine->ar_outputs;
    if (!RunGraph(engine, "decode_ar", 0, *history_len, false,
                   *cache, &outputs, nullptr, nullptr, &ar_input, 0, metrics) ||
        !AppendCacheUpdate(outputs, *history_len, cache, -1, metrics)) {
      return false;
    }
    if (metrics != nullptr) ++metrics->ar_calls;
    ++*history_len;
    pending_ar = &outputs[0];
  }
  if (stop_reason->empty()) *stop_reason = "max_new_tokens";
  return true;
}

/**
 * @brief Generate strictly autoregressively from prefill logits.
 * @param engine Loaded Language runtime state.
 * @param payload Prompt and Vision features.
 * @param max_new_tokens Hard output-token limit.
 * @param prefill_outputs Prefill logits and KV updates.
 * @param cache Mutable committed KV-cache state.
 * @param history_len Mutable committed-token count.
 * @param response Destination generated tokens.
 * @param stop_reason Destination terminal reason.
 * @param metrics Optional generation diagnostics.
 * @param token_callback Optional accepted-token callback.
 * @return True when generation reached a controlled terminal condition.
 */
bool RunArGeneration(EngineState* engine,
                     const InputPayload& payload, int32_t max_new_tokens,
                     const std::vector<rt::Tensor>& prefill_outputs,
                     CacheState* cache, int32_t* history_len,
                     std::vector<int32_t>* response,
                     std::string* stop_reason,
                     GenerationMetrics* metrics,
                     const TokenCallback& token_callback = {}) {
  if (prefill_outputs.empty() || *history_len <= 0) return false;
  const int32_t cache_len = CacheCapacity(*cache);
  if (cache_len <= 0) return false;
  std::vector<int32_t>& generated = engine->generated_tokens;
  generated.assign(payload.prompt_ids.begin(), payload.prompt_ids.end());
  generated.reserve(payload.prompt_ids.size() +
                    static_cast<size_t>(max_new_tokens));
  std::vector<int32_t>& ar_input = engine->ar_input_tokens;
  ar_input.clear();
  ar_input.reserve(1);
  rt::Tensor current_logits;
  const int32_t prefill_logits_row =
      prefill_outputs[0].shape.size() == 3 && prefill_outputs[0].shape[1] == 1
          ? 0
          : (prefill_outputs[0].shape.size() == 3 &&
                     prefill_outputs[0].shape[1] == 7
                 ? 0
                 : prefill_outputs[0].shape[1] - 1);
  if (!SelectLogitsRow(prefill_outputs[0], prefill_logits_row,
                       &current_logits)) {
    return false;
  }

  while (static_cast<int32_t>(response->size()) < max_new_tokens) {
    const int32_t token = rt::DecodeArGreedy(current_logits, generated);
    response->push_back(token);
    generated.push_back(token);
    if (metrics != nullptr) ++metrics->ar_tokens;
    if (token_callback) token_callback(token);
    if (token == kImEndToken) {
      *stop_reason = "im_end";
      return true;
    }
    if (*history_len >= cache_len) {
      *stop_reason = "cache_limit";
      return true;
    }
    ar_input.clear();
    ar_input.push_back(token);
    std::vector<rt::Tensor>& outputs = engine->ar_outputs;
    if (!RunGraph(engine, "decode_ar", 0, *history_len, false,
                   *cache, &outputs, nullptr, nullptr, &ar_input, 0, metrics) ||
        !AppendCacheUpdate(outputs, *history_len, cache, -1, metrics)) {
      return false;
    }
    if (metrics != nullptr) ++metrics->ar_calls;
    ++*history_len;
    current_logits = std::move(outputs[0]);
  }
  *stop_reason = "max_new_tokens";
  return true;
}

/**
 * @brief Run prefill, seed KV state, and dispatch the configured decoder.
 * @param engine Loaded Language runtime state.
 * @param payload Prompt and Vision features.
 * @param max_new_tokens Hard output-token limit.
 * @param generation_mode Requested 'hybrid' or 'slow' mode.
 * @param response Destination generated tokens.
 * @param stop_reason Destination terminal reason.
 * @param metrics Optional generation diagnostics.
 * @param protect_detection_structure Enable guarded detection fallback.
 * @param executed_mode Destination actual mode after fallback.
 * @param fallback_reason Destination reason for switching to slow mode.
 * @param token_callback Optional accepted-token callback.
 * @return True when prefill and generation complete successfully.
 */
bool RunPayload(EngineState* engine,
                 const InputPayload& payload, int32_t max_new_tokens,
                 const std::string& generation_mode,
                 std::vector<int32_t>* response,
                 std::string* stop_reason, GenerationMetrics* metrics,
                 bool protect_detection_structure,
                 std::string* executed_mode,
                 std::string* fallback_reason,
                 const TokenCallback& token_callback = {}) {
  if (engine == nullptr || response == nullptr) return false;
  rt::Graph* prefill = engine->session.GetGraph("prefill");
  if (prefill == nullptr) return false;
  const bool prefill_supports_fused_pbd =
      !prefill->GetOutputShapes().empty() &&
      prefill->GetOutputShapes()[0].size() == 3 &&
      prefill->GetOutputShapes()[0][1] == 7;
  const bool fused_initial_pbd =
      prefill_supports_fused_pbd && payload.prompt_ids.size() + 6 <=
          static_cast<size_t>(prefill->GetInputShapes()[0][1]);

  CacheState& prefill_cache = engine->prefill_cache;
  const auto cache_initialize_started = std::chrono::steady_clock::now();
  if (!BuildZeroCaches(*prefill, &prefill_cache)) return false;
  const double cache_initialize_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cache_initialize_started).count();
  if (metrics != nullptr) metrics->cache_initialize_ms += cache_initialize_ms;

  std::vector<rt::Tensor>& prefill_outputs = engine->prefill_outputs;
  int32_t active_len = prefill->GetInputShapes()[0][1];
  const auto prefill_started = std::chrono::steady_clock::now();
  if (!RunGraph(engine, "prefill", 0, 0, false, prefill_cache,
                &prefill_outputs, &payload, &active_len, nullptr, 0,
                metrics)) {
    return false;
  }
  const double prefill_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - prefill_started).count();
  const int32_t prefill_tokens = active_len;

  CacheState& full_cache = engine->full_cache;
  const auto cache_seed_started = std::chrono::steady_clock::now();
  if (!BuildFullCaches(*prefill, prefill_outputs, active_len, &full_cache)) {
    return false;
  }
  const double cache_seed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - cache_seed_started).count();
  if (metrics != nullptr) metrics->cache_seed_ms += cache_seed_ms;
  if (std::getenv("LA_PROFILE_EXECUTION") != nullptr) {
    std::printf("[profile] cache_initialize=%.3f cache_seed=%.3f\n",
                cache_initialize_ms, cache_seed_ms);
  }

  response->clear();
  const auto decode_started = std::chrono::steady_clock::now();
  std::string selected_mode = generation_mode;
  std::string selected_fallback_reason;
  const TokenCallback generation_callback =
      protect_detection_structure && generation_mode == "hybrid"
          ? TokenCallback{}
          : token_callback;
  const bool generated = generation_mode == "slow"
      ? RunArGeneration(engine, payload, max_new_tokens,
                        prefill_outputs, &full_cache, &active_len, response,
                        stop_reason, metrics, token_callback)
      : RunHybridGeneration(engine, payload, max_new_tokens,
                             &full_cache, &active_len, response, stop_reason,
                             protect_detection_structure,
                             metrics,
                             fused_initial_pbd ? &prefill_outputs[0] : nullptr,
                             generation_callback);
  bool final_generated = generated;
  if (protect_detection_structure && generation_mode == "hybrid" &&
      (!generated || *stop_reason != "im_end")) {
    selected_fallback_reason = stop_reason->empty() ? "hybrid_failed" : *stop_reason;
    response->clear();
    *stop_reason = {};
    active_len = prefill_tokens;
    const auto fallback_cache_seed_started = std::chrono::steady_clock::now();
    if (!BuildFullCaches(*prefill, prefill_outputs, active_len, &full_cache)) {
      return false;
    }
    if (metrics != nullptr) {
      metrics->cache_seed_ms += std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - fallback_cache_seed_started).count();
    }
    selected_mode = "slow";
    final_generated = RunArGeneration(engine, payload, max_new_tokens,
                                       prefill_outputs, &full_cache, &active_len,
                                       response, stop_reason, metrics,
                                       token_callback);
  } else if (protect_detection_structure && generation_mode == "hybrid") {
    EmitTokens(token_callback, *response, response->size());
  }
  if (!final_generated) return false;
  const double decode_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - decode_started).count();
  if (executed_mode != nullptr) *executed_mode = selected_mode;
  if (fallback_reason != nullptr) *fallback_reason = selected_fallback_reason;

  if (metrics != nullptr) {
    metrics->prefill_tokens = prefill_tokens;
    metrics->prefill_ms = prefill_ms;
    metrics->decode_tokens = static_cast<int32_t>(response->size());
    metrics->decode_ms = decode_ms;
  }
  return true;
}

}  // namespace


namespace locateanything {

struct LanguageEngine::Impl {
  EngineState engine;
  std::mutex mutex;
  bool initialized = false;
};

LanguageEngine::LanguageEngine() : impl_(std::make_unique<Impl>()) {}
LanguageEngine::~LanguageEngine() = default;
LanguageEngine::LanguageEngine(LanguageEngine&&) noexcept = default;
LanguageEngine& LanguageEngine::operator=(LanguageEngine&&) noexcept = default;

void LanguageEngine::Initialize(const std::string& model_path,
                                const std::string& embeddings_path,
                                uint32_t backend_mask) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (impl_->initialized) return;
  if (model_path.empty() || embeddings_path.empty()) {
    throw std::invalid_argument("Language model or embeddings path is empty");
  }

  impl_->engine.session.SetBackendMask(backend_mask);
  const rt::Result loaded = impl_->engine.session.Load(model_path);
  if (!loaded.ok()) {
    throw std::runtime_error("cannot load Language HBM: " + loaded.message);
  }
  const rt::GraphValidation validation =
      rt::ValidateLanguageGraphs(impl_->engine.session.GetGraphNames());
  if (!validation.ok()) {
    throw std::runtime_error("Language HBM graph contract mismatch");
  }
  const LanguageAbiProfile abi = ValidateLanguageAbi(&impl_->engine.session);
  std::printf("[Language] ABI graphs=13 prefill=%d cache=%d "
              "fused_prefill=%s compact_logits=%s logits=%d/%s/%s\n",
              abi.prefill_len, abi.cache_len,
              abi.fused_prefill ? "true" : "false",
              abi.compact_logits ? "true" : "false",
              abi.fused_prefill ? 7 : 1,
              abi.compact_logits ? "6" : "query",
              abi.compact_logits ? "1" : "query");
  if (!impl_->engine.embed.Open(embeddings_path, kVocab, kHidden)) {
    throw std::runtime_error("cannot open Language embeddings");
  }

  rt::Graph* prefill = impl_->engine.session.GetGraph("prefill");
  rt::Graph* decode = impl_->engine.session.GetGraph("decode");
  rt::Graph* decode_ar = impl_->engine.session.GetGraph("decode_ar");
  if (prefill == nullptr || decode == nullptr || decode_ar == nullptr) {
    throw std::runtime_error("invalid Language HBM graph interface");
  }
  impl_->initialized = true;
}

LanguageResult LanguageEngine::Generate(
    const LanguageInput& input, int32_t max_new_tokens,
    const std::string& generation_mode,
    bool protect_detection_structure) {
  std::lock_guard<std::mutex> lock(impl_->mutex);
  if (!impl_->initialized) {
    throw std::logic_error("Language engine is not initialized");
  }
  if (max_new_tokens <= 0 ||
      (generation_mode != "hybrid" && generation_mode != "slow")) {
    throw std::invalid_argument("invalid Language generation configuration");
  }
  if (input.prompt_ids.empty() ||
      input.visual_features_fp16.size() %
              (static_cast<size_t>(kHidden) * sizeof(uint16_t)) !=
          0) {
    throw std::invalid_argument("invalid Language input payload");
  }
  size_t image_count = 0;
  for (int32_t token : input.prompt_ids) {
    if (token < 0 || token >= kVocab) {
      throw std::invalid_argument("Language prompt contains an invalid token");
    }
    image_count += token == kImageToken;
  }
  const size_t expected_visual_bytes =
      image_count * static_cast<size_t>(kHidden) * sizeof(uint16_t);
  if (image_count == 0 ||
      input.visual_features_fp16.size() != expected_visual_bytes) {
    throw std::invalid_argument(
        "Language visual features do not match image tokens");
  }
  rt::Graph* prefill = impl_->engine.session.GetGraph("prefill");
  if (prefill == nullptr || prefill->GetInputShapes().empty() ||
      prefill->GetInputShapes()[0].size() != 3) {
    throw std::runtime_error("Language HBM has no valid Prefill graph");
  }
  const int32_t prefill_capacity = prefill->GetInputShapes()[0][1];
  if (input.prompt_ids.size() > static_cast<size_t>(prefill_capacity)) {
    throw std::invalid_argument(
        "Language prompt has " + std::to_string(input.prompt_ids.size()) +
        " tokens; HBM Prefill capacity is " +
        std::to_string(prefill_capacity));
  }

  const int32_t prompt_tokens =
      static_cast<int32_t>(input.prompt_ids.size());
  InputPayload payload{input.prompt_ids, input.visual_features_fp16};
  GenerationMetrics runtime_metrics;
  LanguageResult result;
  std::string executed_mode;
  std::string fallback_reason;
  if (!RunPayload(&impl_->engine, payload, max_new_tokens, generation_mode,
                  &result.token_ids, &result.stop_reason, &runtime_metrics,
                  protect_detection_structure, &executed_mode,
                  &fallback_reason)) {
    throw std::runtime_error("Language generation failed");
  }

  result.metrics.prompt_tokens = prompt_tokens;
  result.metrics.generated_tokens =
      static_cast<int32_t>(result.token_ids.size());
  result.metrics.pbd_calls = runtime_metrics.pbd_calls;
  result.metrics.pbd_accepted_tokens = runtime_metrics.pbd_accepted_tokens;
  result.metrics.ar_calls = runtime_metrics.ar_calls;
  result.metrics.ar_tokens = runtime_metrics.ar_tokens;
  result.metrics.prefill_ms = runtime_metrics.prefill_ms;
  result.metrics.decode_ms = runtime_metrics.decode_ms;
  result.metrics.cache_initialize_ms = runtime_metrics.cache_initialize_ms;
  result.metrics.cache_seed_ms = runtime_metrics.cache_seed_ms;
  result.metrics.cache_update_ms = runtime_metrics.cache_update_ms;
  result.metrics.host_decode_ms = runtime_metrics.host_decode_ms;
  result.metrics.executed_mode = std::move(executed_mode);
  result.metrics.fallback_reason = std::move(fallback_reason);
  result.metrics.graph_timings.reserve(runtime_metrics.graph_timings.size());
  for (const auto& entry : runtime_metrics.graph_timings) {
    GraphTiming timing;
    timing.graph = entry.first;
    timing.calls = entry.second.calls;
    timing.total_ms = entry.second.total_ms;
    timing.input_build_ms = entry.second.input_build_ms;
    timing.buffer_prepare_ms = entry.second.buffer_prepare_ms;
    timing.input_pack_ms = entry.second.input_pack_ms;
    timing.input_flush_ms = entry.second.input_flush_ms;
    timing.bpu_wait_ms = entry.second.bpu_wait_ms;
    timing.submit_ms = entry.second.submit_ms;
    timing.output_flush_ms = entry.second.output_flush_ms;
    timing.output_unpack_ms = entry.second.output_unpack_ms;
    timing.input_bytes = entry.second.input_bytes;
    timing.resident_input_bytes = entry.second.resident_input_bytes;
    timing.output_bytes = entry.second.output_bytes;
    result.metrics.graph_timings.push_back(std::move(timing));
  }
  result.metrics.decode_events.reserve(runtime_metrics.decode_events.size());
  for (const auto& entry : runtime_metrics.decode_events) {
    DecodeEventCount event;
    event.event = entry.first;
    event.count = entry.second;
    result.metrics.decode_events.push_back(std::move(event));
  }
  return result;
}

}  // namespace locateanything

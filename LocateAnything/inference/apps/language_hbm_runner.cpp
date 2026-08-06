// LocateAnything Language HBM runner.
//
// Executes the compiled fixed graphs directly:
//   prefill (q=1024) -> decode (q=6) / decode_ar (q=1)
// It supports both Hybrid PBD and full autoregressive generation.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <vector>

#include "attention_mask.hpp"
#include "embed_lookup.hpp"
#include "language_graph_set.hpp"
#include "hbm_session.hpp"
#include "hybrid_decoder.hpp"
#include "kv_cache_ring.hpp"

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

int64_t ElementCount(const std::vector<int32_t>& shape) {
  int64_t count = 1;
  for (int32_t value : shape) count *= value;
  return count;
}

bool SameShape(const std::vector<int32_t>& left,
               std::initializer_list<int32_t> right) {
  return left == std::vector<int32_t>(right);
}

void PrintShape(const std::vector<int32_t>& shape) {
  std::printf("[");
  for (size_t index = 0; index < shape.size(); ++index) {
    std::printf("%d%s", shape[index], index + 1 == shape.size() ? "" : ",");
  }
  std::printf("]");
}

float Fp16ToFloat(uint16_t bits) {
  const uint32_t sign = (bits >> 15) & 1u;
  const uint32_t exponent = (bits >> 10) & 0x1fu;
  const uint32_t mantissa = bits & 0x3ffu;
  if (exponent == 0) {
    if (mantissa == 0) return sign ? -0.0f : 0.0f;
    const float value = (mantissa / 1024.0f) * 0.00006103515625f;
    return sign ? -value : value;
  }
  if (exponent == 31) return std::numeric_limits<float>::quiet_NaN();
  const float value = std::ldexp(1.0f + mantissa / 1024.0f,
                                 static_cast<int>(exponent) - 15);
  return sign ? -value : value;
}

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

size_t CacheOutputOffset(const std::vector<rt::Tensor>& outputs) {
  return !outputs.empty() && outputs[0].dtype == kS32 &&
                 SameShape(outputs[0].shape, {1, 6, 1})
             ? 7
             : 1;
}

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

bool DumpGraphDebug(const std::string& graph_name, int32_t token_base,
                    int32_t past_len, bool pbd, int32_t pbd_prefix_len,
                    const std::vector<int32_t>* explicit_tokens,
                    const CacheState& cache,
                    const std::vector<rt::Tensor>& outputs) {
  const char* raw_dir = std::getenv("LA_GRAPH_DUMP_DIR");
  if (raw_dir == nullptr || raw_dir[0] == '\0') return true;
  if (outputs.empty()) return false;

  static uint64_t invocation = 0;
  const uint64_t current = ++invocation;
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
  std::vector<int32_t> prompt_ids;
  std::vector<uint8_t> visual_features;
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
  struct GraphTiming {
    int32_t calls = 0;
    double total_ms = 0.0;
    double bpu_wait_ms = 0.0;
    double submit_ms = 0.0;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
  };
  std::map<std::string, GraphTiming> graph_timings;
  double cache_update_ms = 0.0;
  double host_decode_ms = 0.0;
};

using TokenCallback = std::function<void(int32_t)>;

void EmitTokens(const TokenCallback& callback,
                const std::vector<int32_t>& tokens,
                size_t count) {
  if (!callback) return;
  for (size_t index = 0; index < std::min(count, tokens.size()); ++index) {
    callback(tokens[index]);
  }
}

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

bool HasRepeatedDetectionBox(const std::vector<int32_t>& response,
                             const std::vector<int32_t>& candidate) {
  if (!IsCompleteDetectionBox(candidate) || response.size() < candidate.size()) {
    return false;
  }
  return std::search(response.begin(), response.end(), candidate.begin(),
                     candidate.end()) != response.end();
}

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

std::vector<std::string> SplitTabs(const std::string& value) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (true) {
    const size_t separator = value.find('\t', start);
    fields.push_back(value.substr(start, separator - start));
    if (separator == std::string::npos) return fields;
    start = separator + 1;
  }
}

std::string ProtocolText(std::string value) {
  for (char& item : value) {
    if (item == '\t' || item == '\r' || item == '\n') item = ' ';
  }
  return value;
}

bool ReadBinary(const std::string& path, std::vector<uint8_t>* data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const std::streamoff size = stream.tellg();
  if (size < 0) return false;
  data->resize(static_cast<size_t>(size));
  stream.seekg(0);
  return static_cast<bool>(stream.read(
      reinterpret_cast<char*>(data->data()), static_cast<std::streamsize>(size)));
}

bool LoadPayload(const std::string& token_path, const std::string& visual_path,
                 InputPayload* payload) {
  std::vector<uint8_t> token_bytes;
  if (!ReadBinary(token_path, &token_bytes) || token_bytes.empty() ||
      token_bytes.size() % sizeof(int32_t) != 0) {
    return false;
  }
  payload->prompt_ids.resize(token_bytes.size() / sizeof(int32_t));
  std::memcpy(payload->prompt_ids.data(), token_bytes.data(), token_bytes.size());
  for (int32_t token : payload->prompt_ids) {
    if (token < 0 || token >= kVocab) return false;
  }
  if (!ReadBinary(visual_path, &payload->visual_features) ||
      payload->visual_features.size() % (kHidden * sizeof(uint16_t)) != 0) {
    return false;
  }
  size_t image_count = 0;
  for (int32_t token : payload->prompt_ids) image_count += token == kImageToken;
  if (payload->visual_features.size() !=
      image_count * static_cast<size_t>(kHidden) * sizeof(uint16_t)) {
    return false;
  }
  return image_count > 0;
}

void PrintGraphMetadata(const rt::Graph& graph, const std::string& name) {
  std::printf("[graph:%s] inputs=%zu outputs=%zu\n", name.c_str(),
              graph.GetInputNames().size(), graph.GetOutputNames().size());
  for (size_t index = 0; index < graph.GetInputNames().size(); ++index) {
    std::printf("  in[%zu] %s shape=", index,
                graph.GetInputNames()[index].c_str());
    PrintShape(graph.GetInputShapes()[index]);
    std::printf(" dtype=%s\n", rt::DtypeName(graph.GetInputDtypes()[index]));
  }
  for (size_t index = 0; index < graph.GetOutputNames().size(); ++index) {
    std::printf("  out[%zu] %s shape=", index,
                graph.GetOutputNames()[index].c_str());
    PrintShape(graph.GetOutputShapes()[index]);
    std::printf(" dtype=%s\n", rt::DtypeName(graph.GetOutputDtypes()[index]));
  }
}

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

bool BuildPrefillEmbeddings(const rt::Graph& graph, const rt::EmbedLookup& embed,
                            const InputPayload* payload, rt::Tensor* output,
                            int32_t* active_len) {
  const auto& shape = graph.GetInputShapes()[0];
  if (graph.GetInputDtypes()[0] != kF16 || shape.size() != 3 ||
      shape[0] != 1 || shape[1] < 128 || shape[2] != kHidden) {
    return false;
  }
  const int32_t query = shape[1];
  output->shape = shape;
  output->dtype = kF16;
  output->data.assign(static_cast<size_t>(ElementCount(shape)) * sizeof(uint16_t), 0);
  if (payload == nullptr) {
    std::vector<int32_t> token_ids(static_cast<size_t>(query));
    for (int32_t index = 0; index < query; ++index) token_ids[index] = index % kVocab;
    embed.Gather(token_ids.data(), query, output->data.data());
    *active_len = query;
    return true;
  }
  const int32_t length = static_cast<int32_t>(payload->prompt_ids.size());
  if (length <= 0 || length > query) return false;
  const int32_t row_offset = shape[1] - length;
  std::vector<uint8_t> text_embeddings(static_cast<size_t>(length) * kHidden * 2);
  embed.Gather(payload->prompt_ids.data(), length, text_embeddings.data());
  std::memcpy(output->data.data() +
                  static_cast<size_t>(row_offset) * kHidden * sizeof(uint16_t),
              text_embeddings.data(), text_embeddings.size());
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
  std::vector<int32_t> ids(static_cast<size_t>(query), kTextMaskToken);
  ids[0] = payload->prompt_ids.back();
  output->shape = shape;
  output->dtype = kF16;
  output->data.resize(static_cast<size_t>(ElementCount(shape)) * sizeof(uint16_t));
  embed.Gather(ids.data(), query, output->data.data());
  return true;
}

bool BuildPositions(const rt::Graph& graph, int32_t start, bool pbd,
                    int32_t active_len, int32_t pbd_prefix_len,
                    rt::Tensor* output) {
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
    if (active_len > query || start != 0 || pbd) return false;
    const int32_t row_offset = query - active_len;
    for (int32_t index = 0; index < query; ++index) {
      values[index] = index < row_offset ? 0 : index - row_offset;
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

bool BuildMask(const rt::Graph& graph, int32_t past_len, int32_t block_size,
               int32_t active_len, rt::Tensor* output) {
  const auto& shape = graph.GetInputShapes()[2];
  if (graph.GetInputDtypes()[2] != kF16 || shape.size() != 3 || shape[0] != 1) {
    return false;
  }
  const int32_t query = shape[1];
  const int32_t cache_len = shape[2];
  rt::AttentionMask mask;
  if (query >= 128 && active_len >= 0) {
    if (active_len > query || past_len != 0) return false;
    mask.shape = {1, query, cache_len};
    mask.data.assign(static_cast<size_t>(query) * cache_len, kMaskValue);
    const int32_t current_start = cache_len - query;
    const int32_t row_offset = query - active_len;
    for (int32_t row_index = 0; row_index < query; ++row_index) {
      uint16_t* row = mask.data.data() + static_cast<size_t>(row_index) * cache_len;
      if (row_index < row_offset) {
        row[current_start + row_index] = 0;
        continue;
      }
      for (int32_t index = row_offset; index <= row_index; ++index) {
        row[current_start + index] = 0;
      }
    }
  } else if (!rt::BuildAttentionMask(query, cache_len, past_len, block_size,
                                     kMaskValue, false, &mask)) {
    return false;
  }
  output->shape = shape;
  output->dtype = kF16;
  output->data.resize(mask.data.size() * sizeof(uint16_t));
  std::memcpy(output->data.data(), mask.data.data(), output->data.size());
  return true;
}

bool BuildZeroCaches(const rt::Graph& graph, CacheState* state) {
  const auto& shapes = graph.GetInputShapes();
  const auto& dtypes = graph.GetInputDtypes();
  if (shapes.size() != 3 + kCacheCount) return false;
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
  state->tensors.clear();
  state->tensors.reserve(kCacheCount);
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
    cache.shape = input_shapes[input_index];
    cache.dtype = input_dtypes[input_index];
    const size_t cache_bytes = static_cast<size_t>(ElementCount(cache.shape)) *
                               static_cast<size_t>(element_bytes);
    if (!rt::AllocateDeviceBuffer(cache_bytes * 2, true,
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
    state->tensors.push_back(std::move(cache));
  }
  return true;
}

bool BuildInputs(const rt::Graph& graph, const rt::EmbedLookup& embed,
                  int32_t token_base, int32_t past_len, bool pbd,
                  const CacheState& cache, const InputPayload* payload,
                  const std::vector<int32_t>* explicit_tokens,
                  int32_t* active_len, PreparedInputs* inputs,
                  int32_t pbd_prefix_len,
                  const std::vector<int32_t>* generated_tokens = nullptr) {
  if (cache.tensors.size() != kCacheCount) return false;
  const int32_t query = graph.GetInputShapes()[0][1];
  const bool embedding_ok =
      (explicit_tokens != nullptr
           ? BuildExplicitEmbeddings(graph, embed, *explicit_tokens,
                                     &inputs->embeddings)
           : payload != nullptr && query >= 128
           ? BuildPrefillEmbeddings(graph, embed, payload, &inputs->embeddings,
                                    active_len)
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
                      &inputs->positions) ||
      !BuildMask(graph, past_len, pbd ? 6 : 0,
                 payload != nullptr && graph.GetInputShapes()[0][1] >= 128
                     ? *active_len
                     : -1,
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
    inputs->history_mask.data.assign(
        static_cast<size_t>(ElementCount(inputs->history_mask.shape)), 0);
    for (int32_t token : *generated_tokens) {
      if (token < 0 || token >= kVocab) continue;
      for (int32_t row = 0; row < 6; ++row) {
        inputs->history_mask.data[static_cast<size_t>(row) * kVocab + token] = 1;
      }
    }
    inputs->random_values.shape = graph.GetInputShapes()[random_index];
    inputs->random_values.dtype = graph.GetInputDtypes()[random_index];
    inputs->random_values.data.assign(
        static_cast<size_t>(ElementCount(inputs->random_values.shape)) * 2, 0);
    auto* values = reinterpret_cast<uint16_t*>(inputs->random_values.data.data());
    static uint32_t state = 0x9e3779b9u;
    for (int32_t row = 0; row < 6; ++row) {
      state ^= state << 13;
      state ^= state >> 17;
      state ^= state << 5;
      const float uniform = static_cast<float>(state & 0x3ffu) / 1024.0f;
      values[row] = FloatToFp16(uniform);
    }
    inputs->views.push_back(&inputs->history_mask);
    inputs->views.push_back(&inputs->random_values);
  }
  return true;
}

bool PrintLogitsSummary(const std::vector<rt::Tensor>& outputs,
                        const std::string& graph_name) {
  if (CacheOutputOffset(outputs) == 7) return true;
  if (outputs.size() != 1 + kCacheCount || outputs[0].dtype != kF16) {
    std::fprintf(stderr, "[FAIL] %s: unexpected output contract\n",
                 graph_name.c_str());
    return false;
  }
  const rt::Tensor& logits = outputs[0];
  const int64_t values = ElementCount(logits.shape);
  if (values <= 0 || logits.data.size() != static_cast<size_t>(values) * 2) {
    std::fprintf(stderr, "[FAIL] %s: invalid logits buffer\n", graph_name.c_str());
    return false;
  }
  const auto* raw = reinterpret_cast<const uint16_t*>(logits.data.data());
  float min_value = std::numeric_limits<float>::infinity();
  float max_value = -std::numeric_limits<float>::infinity();
  double sum = 0.0;
  int64_t finite = 0;
  int64_t nan_count = 0;
  int32_t argmax = 0;
  float argmax_value = -std::numeric_limits<float>::infinity();
  for (int64_t index = 0; index < values; ++index) {
    const float value = Fp16ToFloat(raw[index]);
    if (value != value) {
      ++nan_count;
      continue;
    }
    min_value = std::min(min_value, value);
    max_value = std::max(max_value, value);
    sum += value;
    ++finite;
    if (logits.shape.size() == 3 && logits.shape[2] == kVocab &&
        index >= static_cast<int64_t>(logits.shape[1] - 1) * kVocab &&
        index < static_cast<int64_t>(logits.shape[1]) * kVocab &&
        value > argmax_value) {
      argmax_value = value;
      argmax = static_cast<int32_t>(index % kVocab);
    }
  }
  std::printf("[%s] logits shape=", graph_name.c_str());
  PrintShape(logits.shape);
  std::printf(" dtype=%s min=%.6f max=%.6f mean=%.6f finite=%lld nan=%lld "
              "last_argmax=%d last_max=%.6f\n",
              rt::DtypeName(logits.dtype), min_value, max_value,
              finite ? static_cast<float>(sum / finite) : 0.0f,
              static_cast<long long>(finite), static_cast<long long>(nan_count),
              argmax, argmax_value);
  return nan_count == 0;
}

void RecordGraphTiming(const std::string& name,
                       const rt::ExecutionMetrics& execution_metrics,
                       GenerationMetrics* metrics) {
  if (metrics == nullptr) return;
  GenerationMetrics::GraphTiming& timing = metrics->graph_timings[name];
  ++timing.calls;
  timing.total_ms += execution_metrics.total_ms;
  timing.bpu_wait_ms += execution_metrics.bpu_wait_ms;
  timing.submit_ms += execution_metrics.submit_ms;
  timing.input_bytes += execution_metrics.input_bytes;
  timing.output_bytes += execution_metrics.output_bytes;
}

bool RunGraph(rt::HbmSession* session, const std::string& name,
              const rt::EmbedLookup& embed, int32_t token_base,
              int32_t past_len, bool pbd, const CacheState& cache,
              std::vector<rt::Tensor>* outputs,
              const InputPayload* payload = nullptr,
              int32_t* active_len = nullptr,
              const std::vector<int32_t>* explicit_tokens = nullptr,
              bool print_summary = true,
              int32_t pbd_prefix_len = 0,
              GenerationMetrics* metrics = nullptr,
              const std::vector<int32_t>* generated_tokens = nullptr) {
  rt::Graph* graph = session->GetGraph(name);
  if (!graph) {
    std::fprintf(stderr, "[FAIL] graph not found: %s\n", name.c_str());
    return false;
  }
  PreparedInputs inputs;
  int32_t local_active_len =
      graph->GetInputShapes()[0].size() > 1 && graph->GetInputShapes()[0][1] >= 128
          ? graph->GetInputShapes()[0][1]
          : -1;
  if (!BuildInputs(*graph, embed, token_base, past_len, pbd, cache, payload,
                   explicit_tokens, &local_active_len, &inputs,
                   pbd_prefix_len, generated_tokens)) {
    std::fprintf(stderr, "[FAIL] cannot build %s inputs\n", name.c_str());
    return false;
  }
  if (active_len != nullptr) *active_len = local_active_len;
  rt::ExecutionMetrics execution_metrics;
  std::vector<rt::OutputSlice> output_slices;
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
      } else if (local_active_len <= output_rows) {
        output_slices[index] =
            rt::OutputSlice{output_rows - local_active_len, local_active_len};
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
  }
  const rt::Result result = session->ExecuteGraphByName(
      name, inputs.views, outputs, &execution_metrics, selected_outputs);
  if (!result.ok()) {
    std::fprintf(stderr, "[FAIL] %s execute code=%d: %s\n", name.c_str(),
                 result.code, result.message.c_str());
    return false;
  }
  RecordGraphTiming(name, execution_metrics, metrics);
  if (!DumpGraphDebug(name, token_base, past_len, pbd, pbd_prefix_len,
                      explicit_tokens, cache, *outputs)) {
    std::fprintf(stderr, "[FAIL] cannot dump debug outputs for %s\n",
                 name.c_str());
    return false;
  }
  if (std::getenv("LA_PROFILE_EXECUTION") != nullptr) {
    std::printf(
        "[profile] graph=%s total=%.3f prepare=%.3f pack=%.3f "
        "input_flush=%.3f submit=%.3f bpu_wait=%.3f output_flush=%.3f "
        "unpack=%.3f input_mib=%.2f resident_input_mib=%.2f output_mib=%.2f\n",
        name.c_str(), execution_metrics.total_ms,
        execution_metrics.buffer_prepare_ms, execution_metrics.input_pack_ms,
        execution_metrics.input_flush_ms, execution_metrics.submit_ms,
        execution_metrics.bpu_wait_ms, execution_metrics.output_flush_ms,
        execution_metrics.output_unpack_ms,
        execution_metrics.input_bytes / (1024.0 * 1024.0),
        execution_metrics.resident_input_bytes / (1024.0 * 1024.0),
        execution_metrics.output_bytes / (1024.0 * 1024.0));
  }
  return !print_summary || PrintLogitsSummary(*outputs, name);
}

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
    const size_t cache_bytes = static_cast<size_t>(cache_len) * row_bytes;
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

int32_t PbdLogitStart(const rt::Tensor& logits, int32_t prefix_len) {
  if (logits.shape.size() != 3 || logits.shape[1] != 6) return prefix_len;
  return 0;
}

int32_t ArLogitRow(const rt::Tensor& logits, int32_t accepted) {
  if (logits.shape.size() != 3 || logits.shape[1] != 1) {
    return accepted - 1;
  }
  return 0;
}

int32_t CacheCapacity(const CacheState& cache);

bool RunHybridGenerationBase(rt::HbmSession* session,
                               const rt::EmbedLookup& embed,
                               const InputPayload& payload,
                               int32_t max_new_tokens, CacheState* cache,
                               int32_t* history_len,
                               std::vector<int32_t>* response,
                               std::string* stop_reason,
                               bool protect_detection_structure,
                               GenerationMetrics* metrics,
                               const TokenCallback& token_callback = {}) {
  std::vector<int32_t> generated = payload.prompt_ids;
  bool use_pbd = true;
  std::vector<rt::Tensor> pending_ar;
  int32_t step = 0;
  const int32_t cache_len = CacheCapacity(*cache);
  if (cache_len <= 0) return false;
  while (static_cast<int32_t>(response->size()) < max_new_tokens) {
    if (*history_len >= cache_len) {
      *stop_reason = "cache_limit";
      break;
    }
    if (use_pbd) {
      if (*history_len + 6 > cache_len) {
        *stop_reason = "cache_limit_before_pbd";
        break;
      }
      const int32_t anchor = generated.back();
      const std::vector<int32_t> draft{
          anchor, kTextMaskToken, kTextMaskToken, kTextMaskToken,
          kTextMaskToken, kTextMaskToken};
      std::vector<rt::Tensor> outputs;
      if (!RunGraph(session, "decode", embed, 0, *history_len, true, *cache,
                    &outputs, nullptr, nullptr, &draft, false, 0, metrics,
                    &generated)) {
        return false;
      }
      if (metrics != nullptr) ++metrics->pbd_calls;
      const bool diagnostics_enabled =
          std::getenv("LA_PBD_DIAGNOSTICS") != nullptr;
      rt::PbdDiagnostics diagnostics;
      const auto pbd_decode_started = std::chrono::steady_clock::now();
      const rt::HybridDecision decision =
          CacheOutputOffset(outputs) == 7
              ? rt::DecodePbdCompact(outputs)
              : rt::DecodePbd(outputs[0], generated, rt::PbdDecodeConfig{},
                              diagnostics_enabled ? &diagnostics : nullptr);
      const double pbd_decode_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - pbd_decode_started).count();
      if (metrics != nullptr) metrics->host_decode_ms += pbd_decode_ms;
      std::printf("[hybrid:%03d] mode=pbd_q6 history=%d pattern=%s accepted=%zu\n",
                  step++, *history_len, decision.type.c_str(), decision.tokens.size());
      if (diagnostics_enabled && diagnostics.valid) {
        std::printf(
            "[pbd] sampling=temperature:0.7,top_p:0.9,repetition:1.1 "
            "host_decode_ms=%.3f "
            "retained=%d,%d,%d,%d,%d,%d box_start=%.6f->%.6f "
            "ref_start=%.6f->%.6f end_score=%.6f->%.6f "
            "coord_top=%.6f/%.6f/%.6f/%.6f->%.6f/%.6f/%.6f/%.6f\n",
            pbd_decode_ms, diagnostics.retained_tokens[0], diagnostics.retained_tokens[1],
            diagnostics.retained_tokens[2], diagnostics.retained_tokens[3],
            diagnostics.retained_tokens[4], diagnostics.retained_tokens[5],
            diagnostics.legacy_box_start, diagnostics.official_box_start,
            diagnostics.legacy_ref_start, diagnostics.official_ref_start,
            diagnostics.legacy_end_score, diagnostics.official_end_score,
            diagnostics.legacy_coord_top[0], diagnostics.legacy_coord_top[1],
            diagnostics.legacy_coord_top[2], diagnostics.legacy_coord_top[3],
            diagnostics.official_coord_top[0], diagnostics.official_coord_top[1],
            diagnostics.official_coord_top[2], diagnostics.official_coord_top[3]);
      }
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

      std::vector<int32_t> commit_tokens(6, decision.tokens.back());
      std::copy(decision.tokens.begin(), decision.tokens.end(), commit_tokens.begin());
      std::vector<rt::Tensor> committed;
      if (!RunGraph(session, "decode", embed, 0, *history_len, false,
                    *cache, &committed, nullptr, nullptr, &commit_tokens, false,
                    0, metrics, &generated) ||
          !AppendCacheUpdate(committed, *history_len, cache, accepted, metrics)) {
        return false;
      }
      if (metrics != nullptr) {
        ++metrics->pbd_calls;
        metrics->pbd_accepted_tokens += accepted;
      }
      *history_len += accepted;
      response->insert(response->end(), decision.tokens.begin(), decision.tokens.end());
      generated.insert(generated.end(), decision.tokens.begin(), decision.tokens.end());
      EmitTokens(token_callback, decision.tokens, decision.tokens.size());
      use_pbd = !decision.switch_to_ar;
      pending_ar.clear();
      if (!use_pbd) {
        rt::Tensor next_logits;
        if (!SelectLogitsRow(committed[0], accepted - 1, &next_logits)) return false;
        pending_ar.push_back(std::move(next_logits));
      }
      continue;
    }

    if (pending_ar.empty()) return false;
    const int32_t token = rt::DecodeArGreedy(pending_ar[0], generated);
    std::printf("[hybrid:%03d] mode=ar_q1 history=%d token=%d\n",
                step++, *history_len, token);
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
    const std::vector<int32_t> one{token};
    std::vector<rt::Tensor> outputs;
    if (!RunGraph(session, "decode_ar", embed, 0, *history_len, false,
                  *cache, &outputs, nullptr, nullptr, &one, false, 0, metrics) ||
        !AppendCacheUpdate(outputs, *history_len, cache, -1, metrics)) {
      return false;
    }
    if (metrics != nullptr) ++metrics->ar_calls;
    ++*history_len;
    pending_ar = std::move(outputs);
    if (token == kBoxEndToken) {
      use_pbd = true;
      pending_ar.clear();
    }
  }
  if (stop_reason->empty()) *stop_reason = "max_new_tokens";
  return true;
}

std::string PbdGraphName(int32_t prefix_len) {
  return prefix_len == 0 ? "decode"
                         : "decode_pbd_q" + std::to_string(6 + prefix_len);
}

std::string ArGraphName(int32_t q_len) {
  return q_len == 1 ? "decode_ar" : "decode_ar_q" + std::to_string(q_len);
}

int32_t CacheCapacity(const CacheState& cache) {
  if (cache.tensors.empty() || cache.tensors.front().shape.size() < 2) return 0;
  return cache.tensors.front().shape[1];
}

bool RunHybridGenerationFused(rt::HbmSession* session,
                              const rt::EmbedLookup& embed,
                              const InputPayload& payload,
                              int32_t max_new_tokens, CacheState* cache,
                              int32_t* history_len,
                              std::vector<int32_t>* response,
                              std::string* stop_reason,
                              bool protect_detection_structure,
                              GenerationMetrics* metrics,
                              const TokenCallback& token_callback = {}) {
  std::vector<int32_t> generated = payload.prompt_ids;
  std::vector<int32_t> pending_pbd;
  std::vector<rt::Tensor> pending_ar;
  bool use_pbd = true;
  int32_t step = 0;
  const int32_t cache_len = CacheCapacity(*cache);
  if (cache_len <= 0) return false;

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
      std::vector<int32_t> tokens;
      if (pending_pbd.empty()) {
        tokens = {generated.back(), kTextMaskToken, kTextMaskToken,
                  kTextMaskToken, kTextMaskToken, kTextMaskToken};
      } else {
        tokens = pending_pbd;
        tokens.push_back(pending_pbd.back());
        tokens.insert(tokens.end(), 5, kTextMaskToken);
      }
      std::vector<rt::Tensor> outputs;
      if (!RunGraph(session, PbdGraphName(prefix_len), embed, 0, *history_len,
                    true, *cache, &outputs, nullptr, nullptr, &tokens, false,
                    prefix_len, metrics, &generated)) {
        return false;
      }
      if (metrics != nullptr) ++metrics->pbd_calls;
      if (prefix_len > 0) {
        if (!AppendCacheUpdate(outputs, *history_len, cache, prefix_len, metrics)) {
          return false;
        }
        *history_len += prefix_len;
      }
      const int32_t pbd_logit_start = PbdLogitStart(outputs[0], prefix_len);
      const auto pbd_decode_started = std::chrono::steady_clock::now();
      const rt::HybridDecision decision =
          CacheOutputOffset(outputs) == 7
              ? rt::DecodePbdCompact(outputs)
              : rt::DecodePbd(outputs[0], generated, rt::PbdDecodeConfig{},
                              nullptr, pbd_logit_start);
      if (metrics != nullptr) {
        metrics->host_decode_ms += std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - pbd_decode_started).count();
      }
      std::printf("[hybrid:%03d] mode=pbd_q%d history=%d pattern=%s accepted=%zu\n",
                  step++, 6 + prefix_len, *history_len, decision.type.c_str(),
                  decision.tokens.size());
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
        std::vector<rt::Tensor> bridge;
        if (!RunGraph(session, ArGraphName(accepted), embed, 0, *history_len,
                      false, *cache, &bridge, nullptr, nullptr,
                      &decision.tokens, false, 0, metrics) ||
            !AppendCacheUpdate(bridge, *history_len, cache, accepted, metrics)) {
          return false;
        }
        if (metrics != nullptr) ++metrics->ar_calls;
        *history_len += accepted;
        rt::Tensor next_logits;
        if (!SelectLogitsRow(bridge[0], ArLogitRow(bridge[0], accepted),
                             &next_logits)) return false;
        pending_ar = {std::move(next_logits)};
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

    if (pending_ar.empty()) return false;
    const int32_t token = rt::DecodeArGreedy(pending_ar[0], generated);
    std::printf("[hybrid:%03d] mode=ar_q1 history=%d token=%d\n",
                step++, *history_len, token);
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
      pending_ar.clear();
      use_pbd = true;
      continue;
    }
    const std::vector<int32_t> one{token};
    std::vector<rt::Tensor> outputs;
    if (!RunGraph(session, "decode_ar", embed, 0, *history_len, false,
                  *cache, &outputs, nullptr, nullptr, &one, false, 0, metrics) ||
        !AppendCacheUpdate(outputs, *history_len, cache, -1, metrics)) {
      return false;
    }
    if (metrics != nullptr) ++metrics->ar_calls;
    ++*history_len;
    pending_ar = std::move(outputs);
  }
  if (stop_reason->empty()) *stop_reason = "max_new_tokens";
  return true;
}

bool RunHybridGeneration(rt::HbmSession* session, const rt::EmbedLookup& embed,
                         const InputPayload& payload, int32_t max_new_tokens,
                         CacheState* cache, int32_t* history_len,
                         std::vector<int32_t>* response,
                         std::string* stop_reason,
                         bool protect_detection_structure,
                         GenerationMetrics* metrics,
                         const TokenCallback& token_callback = {}) {
  if (rt::HasDefaultLanguageGraphs(session->GetGraphNames())) {
    std::printf("[INFO] Language graph set=fused_decode\n");
    return RunHybridGenerationFused(session, embed, payload, max_new_tokens,
                                    cache, history_len, response, stop_reason,
                                    protect_detection_structure,
                                    metrics,
                                    token_callback);
  }
  std::printf("[INFO] fused_decode extensions unavailable; using base decode path\n");
  return RunHybridGenerationBase(session, embed, payload, max_new_tokens,
                                   cache, history_len, response, stop_reason,
                                   protect_detection_structure,
                                   metrics,
                                   token_callback);
}

bool RunArGeneration(rt::HbmSession* session, const rt::EmbedLookup& embed,
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
  std::vector<int32_t> generated = payload.prompt_ids;
  rt::Tensor current_logits;
  const int32_t prefill_logits_row =
      prefill_outputs[0].shape.size() == 3 && prefill_outputs[0].shape[1] == 1
          ? 0
          : *history_len - 1;
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
    const std::vector<int32_t> one{token};
    std::vector<rt::Tensor> outputs;
    if (!RunGraph(session, "decode_ar", embed, 0, *history_len, false,
                  *cache, &outputs, nullptr, nullptr, &one, false, 0, metrics) ||
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

bool WriteTokenOutput(const std::string& path,
                      const std::vector<int32_t>& response,
                      const std::string& stop_reason,
                      const std::string& executed_mode = {},
                      const std::string& fallback_reason = {}) {
  if (path.empty()) return true;
  std::ofstream stream(path, std::ios::trunc);
  if (!stream) return false;
  stream << "stop_reason=" << stop_reason << '\n';
  stream << "token_ids=";
  for (size_t index = 0; index < response.size(); ++index) {
    if (index) stream << ',';
    stream << response[index];
  }
  stream << '\n' << "structured=" << rt::RenderLocateAnythingTokens(response) << '\n';
  if (!executed_mode.empty()) stream << "executed_mode=" << executed_mode << '\n';
  if (!fallback_reason.empty()) stream << "fallback_reason=" << fallback_reason << '\n';
  return static_cast<bool>(stream);
}

std::string EncodeGraphMetrics(const GenerationMetrics& metrics) {
  std::string encoded;
  for (const auto& entry : metrics.graph_timings) {
    if (!encoded.empty()) encoded.push_back(';');
    const auto& timing = entry.second;
    encoded += entry.first + "," + std::to_string(timing.calls) + "," +
               std::to_string(timing.total_ms) + "," +
               std::to_string(timing.bpu_wait_ms) + "," +
               std::to_string(timing.submit_ms) + "," +
               std::to_string(timing.input_bytes) + "," +
               std::to_string(timing.output_bytes);
  }
  return encoded;
}

bool RunPayload(rt::HbmSession* session, rt::EmbedLookup* embed,
                const InputPayload& payload, int32_t max_new_tokens,
                const std::string& generation_mode,
                const std::string& output_path, std::string* stop_reason,
                size_t* response_size, GenerationMetrics* metrics,
                bool protect_detection_structure,
                std::string* executed_mode,
                std::string* fallback_reason,
                const TokenCallback& token_callback = {}) {
  rt::Graph* prefill = session->GetGraph("prefill");
  if (prefill == nullptr) return false;

  CacheState prefill_cache;
  if (!BuildZeroCaches(*prefill, &prefill_cache)) return false;

  std::vector<rt::Tensor> prefill_outputs;
  int32_t active_len = prefill->GetInputShapes()[0][1];
  const auto prefill_started = std::chrono::steady_clock::now();
  if (!RunGraph(session, "prefill", *embed, 0, 0, false, prefill_cache,
                &prefill_outputs, &payload, &active_len, nullptr, false, 0,
                metrics)) {
    return false;
  }
  const double prefill_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - prefill_started).count();
  const int32_t prefill_tokens = active_len;

  CacheState full_cache;
  if (!BuildFullCaches(*prefill, prefill_outputs, active_len, &full_cache)) {
    return false;
  }

  std::vector<int32_t> response;
  const auto decode_started = std::chrono::steady_clock::now();
  std::string selected_mode = generation_mode;
  std::string selected_fallback_reason;
  const TokenCallback generation_callback =
      protect_detection_structure && generation_mode == "hybrid"
          ? TokenCallback{}
          : token_callback;
  const bool generated = generation_mode == "slow"
      ? RunArGeneration(session, *embed, payload, max_new_tokens,
                        prefill_outputs, &full_cache, &active_len, &response,
                        stop_reason, metrics, token_callback)
      : RunHybridGeneration(session, *embed, payload, max_new_tokens,
                             &full_cache, &active_len, &response, stop_reason,
                            protect_detection_structure,
                            metrics,
                            generation_callback);
  bool final_generated = generated;
  if (protect_detection_structure && generation_mode == "hybrid" &&
      (!generated || *stop_reason != "im_end")) {
    selected_fallback_reason = stop_reason->empty() ? "hybrid_failed" : *stop_reason;
    response.clear();
    *stop_reason = {};
    active_len = prefill_tokens;
    if (!BuildFullCaches(*prefill, prefill_outputs, active_len, &full_cache)) {
      return false;
    }
    selected_mode = "slow";
    final_generated = RunArGeneration(session, *embed, payload, max_new_tokens,
                                       prefill_outputs, &full_cache, &active_len,
                                       &response, stop_reason, metrics,
                                       token_callback);
  } else if (protect_detection_structure && generation_mode == "hybrid") {
    EmitTokens(token_callback, response, response.size());
  }
  if (!final_generated) return false;
  const double decode_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - decode_started).count();
  if (!WriteTokenOutput(output_path, response, *stop_reason, selected_mode,
                        selected_fallback_reason)) return false;

  if (executed_mode != nullptr) *executed_mode = selected_mode;
  if (fallback_reason != nullptr) *fallback_reason = selected_fallback_reason;

  *response_size = response.size();
  if (metrics != nullptr) {
    metrics->prefill_tokens = prefill_tokens;
    metrics->prefill_ms = prefill_ms;
    metrics->decode_tokens = static_cast<int32_t>(response.size());
    metrics->decode_ms = decode_ms;
  }
  std::printf("[generation] requested=%s executed=%s stop=%s tokens=%zu history=%d\n",
              generation_mode.c_str(), selected_mode.c_str(), stop_reason->c_str(), response.size(),
              active_len);
  if (!selected_fallback_reason.empty()) {
    std::printf("[generation] fallback_reason=%s\n",
                selected_fallback_reason.c_str());
  }
  std::printf("[generation] structured=%s\n",
              rt::RenderLocateAnythingTokens(response).c_str());
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::fprintf(stderr,
                 "usage: %s --model LANGUAGE.hbm --embed embed_tokens.bin "
                 "[--mode all|prefill|decode|decode_ar] "
                 "[--tokens prompt.i32.bin --visual visual.f16.bin] "
                 "[--generation-mode hybrid|slow] "
                 "[--max-new-tokens N --output result.txt] "
                 "[--backend-mask MASK] "
                 "[--expected-prefill-chunk N --expected-cache-len N] "
                 "[--structured-output] [--server]\n",
                 argv[0]);
    return 1;
  }
  std::string model_path;
  std::string embed_path;
  std::string token_path;
  std::string visual_path;
  std::string output_path;
  int32_t max_new_tokens = 0;
  std::string mode = "all";
  std::string generation_mode = "hybrid";
  bool server = false;
  bool structured_output = false;
  uint32_t backend_mask = 15;
  int32_t expected_prefill_chunk = 0;
  int32_t expected_cache_len = 0;
  for (int index = 1; index < argc; ++index) {
    const std::string arg = argv[index];
    if (arg == "--model" && index + 1 < argc) model_path = argv[++index];
    else if (arg == "--embed" && index + 1 < argc) embed_path = argv[++index];
    else if (arg == "--tokens" && index + 1 < argc) token_path = argv[++index];
    else if (arg == "--visual" && index + 1 < argc) visual_path = argv[++index];
    else if (arg == "--output" && index + 1 < argc) output_path = argv[++index];
    else if (arg == "--max-new-tokens" && index + 1 < argc) {
      max_new_tokens = std::stoi(argv[++index]);
    }
    else if (arg == "--mode" && index + 1 < argc) mode = argv[++index];
    else if (arg == "--generation-mode" && index + 1 < argc) {
      generation_mode = argv[++index];
    }
    else if (arg == "--backend-mask" && index + 1 < argc) {
      try {
        const unsigned long value = std::stoul(argv[++index], nullptr, 0);
        if (value > 0xFFFFFFFFUL) {
          std::fprintf(stderr, "invalid backend mask\n");
          return 1;
        }
        backend_mask = static_cast<uint32_t>(value);
      } catch (...) {
        std::fprintf(stderr, "invalid backend mask\n");
        return 1;
      }
    }
    else if (arg == "--expected-prefill-chunk" && index + 1 < argc) {
      expected_prefill_chunk = std::stoi(argv[++index]);
    }
    else if (arg == "--expected-cache-len" && index + 1 < argc) {
      expected_cache_len = std::stoi(argv[++index]);
    }
    else if (arg == "--server") {
      server = true;
    }
    else if (arg == "--structured-output") {
      structured_output = true;
    }
    else {
      std::fprintf(stderr, "unknown or incomplete argument: %s\n", arg.c_str());
      return 1;
    }
  }
  if (model_path.empty() || embed_path.empty() ||
      (token_path.empty() != visual_path.empty()) ||
      max_new_tokens < 0 || (max_new_tokens > 0 && token_path.empty()) ||
      expected_prefill_chunk < 0 || expected_cache_len < 0 ||
      (generation_mode != "hybrid" && generation_mode != "slow") ||
      (mode != "all" && mode != "prefill" && mode != "decode" && mode != "decode_ar") ||
      (server && (!token_path.empty() || !visual_path.empty()))) {
    std::fprintf(stderr, "invalid arguments\n");
    return 1;
  }

  rt::HbmSession session;
  session.SetBackendMask(backend_mask);
  const rt::Result loaded = session.Load(model_path);
  if (!loaded.ok()) {
    std::fprintf(stderr, "[FAIL] load: %s\n", loaded.message.c_str());
    return 2;
  }
  std::printf("[ok] loaded graphs:");
  const std::vector<std::string> graph_names = session.GetGraphNames();
  for (const auto& name : graph_names) std::printf(" %s", name.c_str());
  std::printf("\n");
  const rt::GraphSetValidation base_validation =
      rt::ValidateGraphNames(rt::BaseLanguageGraphNames(), graph_names);
  if (!base_validation.ok()) {
    std::fprintf(stderr, "[FAIL] HBM is missing required base Language graphs\n");
    return 4;
  }
  std::printf("[ok] Language graphs=%zu fused_decode=%s\n",
              graph_names.size(),
              rt::HasDefaultLanguageGraphs(graph_names) ? "available" : "not-available");
  for (const auto& name : graph_names) {
    rt::Graph* graph = session.GetGraph(name);
    if (graph) PrintGraphMetadata(*graph, name);
  }

  rt::EmbedLookup embed;
  if (!embed.Open(embed_path, kVocab, kHidden)) {
    std::fprintf(stderr, "[FAIL] open embed_tokens.bin\n");
    return 3;
  }

  rt::Graph* prefill = session.GetGraph("prefill");
  rt::Graph* decode = session.GetGraph("decode");
  rt::Graph* decode_ar = session.GetGraph("decode_ar");
  if (!prefill || !decode || !decode_ar) return 4;
  const auto& prefill_inputs = prefill->GetInputShapes();
  const auto& decode_inputs = decode->GetInputShapes();
  if (prefill_inputs.empty() || decode_inputs.size() <= 3) {
    std::fprintf(stderr, "[FAIL] invalid Language HBM graph interface\n");
    return 4;
  }
  const auto& prefill_shape = prefill_inputs[0];
  const auto& decode_cache_shape = decode_inputs[3];
  if ((expected_prefill_chunk > 0 &&
       (prefill_shape.size() < 2 || prefill_shape[1] != expected_prefill_chunk)) ||
      (expected_cache_len > 0 &&
       (decode_cache_shape.size() < 2 || decode_cache_shape[1] != expected_cache_len))) {
    std::fprintf(stderr,
                 "[FAIL] inference language dimensions do not match the HBM graph interface\n");
    return 4;
  }

  if (server) {
    std::printf("LAHBM/1\tREADY\tlanguage\n");
    std::fflush(stdout);
    std::string request;
    while (std::getline(std::cin, request)) {
      if (request == "LAHBM/1\tQUIT") break;
      const std::vector<std::string> fields = SplitTabs(request);
      if (fields.size() < 8 || fields.size() > 10 ||
          fields[0] != "LAHBM/1" ||
          fields[1] != "RUN" || fields[2].empty()) {
        const char* request_id =
            fields.size() > 2 && !fields[2].empty() ? fields[2].c_str() : "0";
        std::printf("LAHBM/1\tERROR\t%s\t1\tinvalid request frame\n",
                    request_id);
        std::fflush(stdout);
        continue;
      }

      InputPayload request_payload;
      int32_t request_tokens = 0;
      try {
        request_tokens = std::stoi(fields[6]);
      } catch (const std::exception&) {
        request_tokens = 0;
      }
      const std::string request_mode = fields[7];
      const bool protect_detection_structure =
          fields.size() >= 9 && fields[8] == "1";
      const bool stream_tokens = fields.size() < 10 || fields[9] != "0";
      std::string request_stop;
      std::string request_executed_mode;
      std::string request_fallback_reason;
      size_t request_response_size = 0;
      GenerationMetrics request_metrics;
      TokenCallback token_callback;
      if (stream_tokens) {
        token_callback = [&fields](int32_t token) {
          std::printf("LAHBM/1\tTOKEN\t%s\t%d\n", fields[2].c_str(), token);
          std::fflush(stdout);
        };
      }
      bool ok = request_tokens > 0 &&
                (request_mode == "slow" || request_mode == "hybrid") &&
                LoadPayload(fields[3], fields[4], &request_payload) &&
                RunPayload(&session, &embed, request_payload, request_tokens,
                           request_mode, fields[5], &request_stop,
                           &request_response_size, &request_metrics,
                           protect_detection_structure, &request_executed_mode,
                           &request_fallback_reason,
                           token_callback);
      if (!ok) {
        std::printf("LAHBM/1\tERROR\t%s\t1\trequest failed\n",
                    fields[2].c_str());
      } else {
        const std::string graph_metrics = EncodeGraphMetrics(request_metrics);
        std::printf("LAHBM/1\tRESULT\t%s\t%s\t%zu\t%d\t%.3f\t%.3f\t%s\t%s"
                    "\t%d\t%d\t%d\t%d\t%s\t%.3f\t%.3f\n",
                    fields[2].c_str(), request_stop.c_str(), request_response_size,
                    request_metrics.prefill_tokens, request_metrics.prefill_ms,
                    request_metrics.decode_ms,
                    request_executed_mode.empty() ? request_mode.c_str()
                                                  : request_executed_mode.c_str(),
                    request_fallback_reason.c_str(), request_metrics.pbd_calls,
                    request_metrics.pbd_accepted_tokens, request_metrics.ar_calls,
                    request_metrics.ar_tokens, graph_metrics.c_str(),
                    request_metrics.cache_update_ms, request_metrics.host_decode_ms);
      }
      std::fflush(stdout);
    }
    return 0;
  }
  CacheState prefill_cache;
  if (!BuildZeroCaches(*prefill, &prefill_cache)) {
    std::fprintf(stderr, "[FAIL] unsupported prefill cache layout\n");
    return 5;
  }
  InputPayload payload;
  const InputPayload* payload_ptr = nullptr;
  if (!token_path.empty()) {
    if (!LoadPayload(token_path, visual_path, &payload)) {
      std::fprintf(stderr, "[FAIL] invalid payload; tokens must contain the "
                           "prompt IDs and visual must contain one FP16 row per image token\n");
      return 6;
    }
    payload_ptr = &payload;
    std::printf("[ok] payload prompt_tokens=%zu visual_tokens=%zu\n",
                payload.prompt_ids.size(),
                payload.visual_features.size() / (static_cast<size_t>(kHidden) * 2));
  }

  if (max_new_tokens > 0 && (mode == "all" || mode == "prefill")) {
    std::string stop_reason;
    std::string executed_mode;
    std::string fallback_reason;
    size_t response_size = 0;
    GenerationMetrics metrics;
    if (!RunPayload(&session, &embed, payload, max_new_tokens, generation_mode,
                    output_path, &stop_reason, &response_size,
                    &metrics, structured_output, &executed_mode,
                    &fallback_reason)) {
      std::fprintf(stderr, "[FAIL] %s generation failed\n",
                   generation_mode.c_str());
      return 12;
    }
    std::printf("[verdict] language HBM %s generation PASSED\n",
                executed_mode.c_str());
    return 0;
  }

  std::vector<rt::Tensor> outputs;
  int32_t active_len = prefill->GetInputShapes()[0][1];
  bool ok = true;
  if (mode == "all" || mode == "prefill") {
    ok = RunGraph(&session, "prefill", embed, 0, 0, false,
                  prefill_cache, &outputs, payload_ptr, &active_len) && ok;
    if (!ok) return 6;
    CacheState full_cache;
    if (!BuildFullCaches(*prefill, outputs, active_len, &full_cache)) {
      std::fprintf(stderr, "[FAIL] cannot materialize full prefill KV cache\n");
      return 7;
    }
    prefill_cache = std::move(full_cache);
  }
  if (mode == "all" || mode == "decode") {
    if (mode == "decode") {
      ok = RunGraph(&session, "prefill", embed, 0, 0, false,
                    prefill_cache, &outputs, payload_ptr, &active_len) && ok;
      if (!ok) return 7;
      CacheState full_cache;
      if (!BuildFullCaches(*prefill, outputs, active_len, &full_cache)) return 7;
      prefill_cache = std::move(full_cache);
    }
    std::vector<rt::Tensor> decode_outputs;
    ok = RunGraph(&session, "decode", embed, 1, active_len, true,
                  prefill_cache, &decode_outputs, payload_ptr) && ok;
    if (!ok) return 8;
  }
  if (mode == "all" || mode == "decode_ar") {
    if (mode == "decode_ar") {
      ok = RunGraph(&session, "prefill", embed, 0, 0, false,
                    prefill_cache, &outputs, payload_ptr, &active_len) && ok;
      if (!ok) return 9;
      CacheState full_cache;
      if (!BuildFullCaches(*prefill, outputs, active_len, &full_cache)) return 9;
      prefill_cache = std::move(full_cache);
    }
    std::vector<rt::Tensor> ar_outputs;
    ok = RunGraph(&session, "decode_ar", embed, 1, active_len, false,
                  prefill_cache, &ar_outputs, payload_ptr) && ok;
    if (!ok) return 10;
  }
  std::printf("[verdict] language HBM runtime %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 11;
}

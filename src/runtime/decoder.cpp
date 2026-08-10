#include "runtime/decoder.hpp"

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <limits>
#include <sstream>
#include <thread>
#include <utility>

#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
#include <arm_neon.h>
#endif

namespace locateanything_runtime {
namespace {

constexpr int32_t kVocab = 152681;
constexpr int32_t kBoxStart = 151668;
constexpr int32_t kBoxEnd = 151669;
constexpr int32_t kRefStart = 151672;
constexpr int32_t kRefEnd = 151673;
constexpr int32_t kCoordStart = 151677;
constexpr int32_t kCoordEnd = 152677;
constexpr int32_t kTextMask = 151676;
constexpr int32_t kNull = 152678;
constexpr int32_t kImEnd = 151645;
constexpr int32_t kNone = 4064;
constexpr int32_t kNucleusInitialWidth = 256;
constexpr int32_t kNucleusMaximumWidth = 8192;

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

void ConvertFp16Row(const uint16_t *raw, std::vector<float> *values) {
  size_t token = 0;
#if defined(__aarch64__) && defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
  const uint16x8_t exponent_mask = vdupq_n_u16(0x7c00);
  for (; token + 8 <= static_cast<size_t>(kVocab); token += 8) {
    const uint16x8_t bits = vld1q_u16(raw + token);
    const float16x8_t halves = vreinterpretq_f16_u16(bits);
    vst1q_f32(values->data() + token,
              vcvt_f32_f16(vget_low_f16(halves)));
    vst1q_f32(values->data() + token + 4,
              vcvt_f32_f16(vget_high_f16(halves)));

    const uint16x8_t special =
        vceqq_u16(vandq_u16(bits, exponent_mask), exponent_mask);
    if (vmaxvq_u16(special) != 0) {
      for (size_t lane = 0; lane < 8; ++lane) {
        if ((raw[token + lane] & 0x7c00) == 0x7c00) {
          (*values)[token + lane] = std::numeric_limits<float>::quiet_NaN();
        }
      }
    }
  }
#endif
  for (; token < static_cast<size_t>(kVocab); ++token) {
    (*values)[token] = Fp16ToFloat(raw[token]);
  }
}

std::vector<float> Row(const Tensor &logits, int32_t row,
                       const std::vector<int32_t> &history_tokens,
                       float repetition_penalty) {
  const auto *raw = reinterpret_cast<const uint16_t *>(logits.data.data());
  std::vector<float> values(kVocab);
  const size_t offset = static_cast<size_t>(row) * kVocab;
  ConvertFp16Row(raw + offset, &values);
  if (repetition_penalty != 1.0f) {
    for (const int32_t token : history_tokens) {
      float &value = values[static_cast<size_t>(token)];
      value = value > 0 ? value / repetition_penalty
                        : value * repetition_penalty;
    }
  }
  return values;
}

struct NucleusDistribution {
  std::vector<float> probabilities;
  int32_t retained = 0;
};

std::vector<float> Softmax(const std::vector<float> &logits);
int32_t Argmax(const std::vector<float> &values);

std::vector<float> Softmax(const std::vector<float> &logits,
                           float maximum) {
  std::vector<float> probabilities(logits.size());
  double total = 0.0;
  for (size_t index = 0; index < logits.size(); ++index) {
    probabilities[index] = std::exp(logits[index] - maximum);
    total += probabilities[index];
  }
  if (total <= 0.0 || !std::isfinite(total)) return probabilities;
  for (float &value : probabilities) value = static_cast<float>(value / total);
  return probabilities;
}

NucleusDistribution NucleusSoftmax(const std::vector<float> &logits,
                                    float temperature, float top_p) {
  std::vector<float> adjusted(logits.size());
  const float inverse_temperature = 1.0f / temperature;
  float maximum = -std::numeric_limits<float>::infinity();
  float second_maximum = -std::numeric_limits<float>::infinity();
  size_t maximum_token = 0;
  for (size_t index = 0; index < logits.size(); ++index) {
    const float value = logits[index] * inverse_temperature;
    adjusted[index] = value;
    if (value > maximum) {
      second_maximum = maximum;
      maximum = value;
      maximum_token = index;
    } else if (value > second_maximum) {
      second_maximum = value;
    }
  }

  if (top_p < 1.0f && std::isfinite(maximum) &&
      std::isfinite(second_maximum)) {
    const double other_mass_upper_bound =
        static_cast<double>(adjusted.size() - 1) *
        std::exp(static_cast<double>(second_maximum - maximum));
    const double maximum_probability_lower_bound =
        1.0 / (1.0 + other_mass_upper_bound);
    if (maximum_probability_lower_bound >
        static_cast<double>(top_p) + 1e-6) {
      std::vector<float> probabilities(logits.size(), 0.0f);
      probabilities[maximum_token] = 1.0f;
      return {std::move(probabilities), 1};
    }
  }

  std::vector<float> raw = Softmax(adjusted, maximum);
  if (top_p >= 1.0f) {
    return {std::move(raw), static_cast<int32_t>(logits.size())};
  }

  const auto maximum_probability = std::max_element(raw.begin(), raw.end());
  if (maximum_probability != raw.end() && *maximum_probability > top_p) {
    const size_t token = static_cast<size_t>(maximum_probability - raw.begin());
    std::fill(raw.begin(), raw.end(), 0.0f);
    raw[token] = 1.0f;
    return {std::move(raw), 1};
  }

  std::vector<int32_t> indices(logits.size());
  std::iota(indices.begin(), indices.end(), 0);
  auto greater = [&](int32_t left, int32_t right) {
    return adjusted[static_cast<size_t>(left)] >
           adjusted[static_cast<size_t>(right)];
  };
  double cumulative = 0.0;
  size_t retained = 0;
  size_t width = std::min<size_t>(kNucleusInitialWidth, indices.size());
  size_t partitioned = 0;
  while (true) {
    if (width < indices.size()) {
      std::nth_element(indices.begin() + partitioned,
                       indices.begin() + width, indices.end(), greater);
      std::sort(indices.begin(), indices.begin() + width, greater);
    } else {
      std::sort(indices.begin() + partitioned, indices.end(), greater);
    }
    cumulative = 0.0;
    retained = 0;
    for (; retained < width; ++retained) {
      cumulative += raw[static_cast<size_t>(indices[retained])];
      if (cumulative >= top_p) {
        ++retained;
        break;
      }
    }
    if (cumulative >= top_p || width == indices.size()) break;
    partitioned = width;
    if (width >= static_cast<size_t>(kNucleusMaximumWidth)) {
      width = indices.size();
    } else {
      width = std::min({width * 2, indices.size(),
                        static_cast<size_t>(kNucleusMaximumWidth)});
    }
  }

  if (retained == 0 || cumulative <= 0.0 || !std::isfinite(cumulative)) {
    std::fill(raw.begin(), raw.end(), 0.0f);
    return {std::move(raw), 0};
  }
  // Keep the same retained-token values while reusing the full softmax
  // allocation. The decoder only observes retained entries; all others must
  // be zero after nucleus filtering.
  for (size_t index = retained; index < indices.size(); ++index) {
    raw[static_cast<size_t>(indices[index])] = 0.0f;
  }
  for (size_t index = 0; index < retained; ++index) {
    const size_t token = static_cast<size_t>(indices[index]);
    raw[token] = static_cast<float>(raw[token] / cumulative);
  }
  return {std::move(raw), static_cast<int32_t>(retained)};
}

struct PbdRowResult {
  std::vector<float> probabilities;
  std::vector<float> legacy_probabilities;
  int32_t greedy_token = 0;
  int32_t retained_tokens = 0;
};

PbdRowResult DecodePbdRow(const Tensor &logits, int32_t row,
                          const std::vector<int32_t> &history_tokens,
                          const PbdDecodeConfig &config,
                          bool collect_diagnostics) {
  std::vector<float> adjusted =
      Row(logits, row, history_tokens, config.repetition_penalty);
  std::vector<float> legacy;
  if (collect_diagnostics) legacy = Softmax(adjusted);
  NucleusDistribution nucleus =
      NucleusSoftmax(adjusted, config.temperature, config.top_p);
  const int32_t greedy = Argmax(nucleus.probabilities);
  return {std::move(nucleus.probabilities), std::move(legacy), greedy,
          nucleus.retained};
}

class PbdRowExecutor {
 public:
  PbdRowExecutor() {
    workers_.reserve(5);
    for (int32_t row = 0; row < 5; ++row) {
      workers_.emplace_back([this, row] { Worker(row); });
    }
  }

  ~PbdRowExecutor() {
    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      stopping_ = true;
    }
    job_ready_.notify_all();
    for (std::thread &worker : workers_) worker.join();
  }

  PbdRowExecutor(const PbdRowExecutor &) = delete;
  PbdRowExecutor &operator=(const PbdRowExecutor &) = delete;

  void Decode(const Tensor &logits,
              const std::vector<int32_t> &history_tokens,
              const PbdDecodeConfig &config, bool collect_diagnostics,
              int32_t row_start,
              std::array<PbdRowResult, 6> *results) {
    std::lock_guard<std::mutex> call_lock(call_mutex_);
    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      logits_ = &logits;
      history_tokens_ = &history_tokens;
      config_ = config;
      collect_diagnostics_ = collect_diagnostics;
      row_start_ = row_start;
      results_ = results;
      pending_workers_ = 5;
      ++generation_;
    }
    job_ready_.notify_all();

    (*results)[5] =
        DecodePbdRow(logits, row_start + 5, history_tokens, config,
                     collect_diagnostics);

    std::unique_lock<std::mutex> lock(job_mutex_);
    job_done_.wait(lock, [this] { return pending_workers_ == 0; });
    logits_ = nullptr;
    history_tokens_ = nullptr;
    results_ = nullptr;
  }

 private:
  void Worker(int32_t row) {
    size_t observed_generation = 0;
    while (true) {
      const Tensor *logits = nullptr;
      const std::vector<int32_t> *history_tokens = nullptr;
      PbdDecodeConfig config;
      bool collect_diagnostics = false;
      int32_t row_start = 0;
      std::array<PbdRowResult, 6> *results = nullptr;
      {
        std::unique_lock<std::mutex> lock(job_mutex_);
        job_ready_.wait(lock, [this, observed_generation] {
          return stopping_ || generation_ != observed_generation;
        });
        if (stopping_) return;
        observed_generation = generation_;
        logits = logits_;
        history_tokens = history_tokens_;
        config = config_;
        collect_diagnostics = collect_diagnostics_;
        row_start = row_start_;
        results = results_;
      }

      (*results)[static_cast<size_t>(row)] =
          DecodePbdRow(*logits, row_start + row, *history_tokens, config,
                       collect_diagnostics);

      {
        std::lock_guard<std::mutex> lock(job_mutex_);
        --pending_workers_;
        if (pending_workers_ == 0) job_done_.notify_one();
      }
    }
  }

  std::mutex call_mutex_;
  std::mutex job_mutex_;
  std::condition_variable job_ready_;
  std::condition_variable job_done_;
  std::vector<std::thread> workers_;
  const Tensor *logits_ = nullptr;
  const std::vector<int32_t> *history_tokens_ = nullptr;
  PbdDecodeConfig config_;
  std::array<PbdRowResult, 6> *results_ = nullptr;
  size_t generation_ = 0;
  int32_t pending_workers_ = 0;
  int32_t row_start_ = 0;
  bool collect_diagnostics_ = false;
  bool stopping_ = false;
};

PbdRowExecutor &PbdRows() {
  static PbdRowExecutor executor;
  return executor;
}

std::vector<float> Softmax(const std::vector<float> &logits) {
  const float maximum = *std::max_element(logits.begin(), logits.end());
  return Softmax(logits, maximum);
}

std::vector<int32_t> TopK(const std::vector<float> &values, int32_t count) {
  count = std::min<int32_t>(count, static_cast<int32_t>(values.size()));
  if (count <= 0) return {};
  auto better = [&](int32_t left, int32_t right) {
    const float lhs = values[static_cast<size_t>(left)];
    const float rhs = values[static_cast<size_t>(right)];
    return lhs > rhs || (lhs == rhs && left < right);
  };
  std::vector<int32_t> top;
  top.reserve(static_cast<size_t>(count));
  for (int32_t token = 0; token < static_cast<int32_t>(values.size()); ++token) {
    if (static_cast<int32_t>(top.size()) < count) {
      top.push_back(token);
    } else if (!better(token, top.back())) {
      continue;
    } else {
      top.back() = token;
    }
    for (size_t index = top.size() - 1;
         index > 0 && better(top[index], top[index - 1]); --index) {
      std::swap(top[index], top[index - 1]);
    }
  }
  return top;
}

const std::vector<int32_t> &BuildHistoryTokens(
    const std::vector<int32_t> &generated) {
  static thread_local std::vector<uint8_t> seen(
      static_cast<size_t>(kVocab), 0);
  static thread_local std::vector<int32_t> history_tokens;
  for (const int32_t token : history_tokens) {
    seen[static_cast<size_t>(token)] = 0;
  }
  history_tokens.clear();
  for (int32_t token : generated) {
    if (token >= 0 && token < kVocab &&
        seen[static_cast<size_t>(token)] == 0) {
      seen[static_cast<size_t>(token)] = 1;
      history_tokens.push_back(token);
    }
  }
  return history_tokens;
}

int32_t Argmax(const std::vector<float> &values) {
  return static_cast<int32_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

float EndScore(const std::vector<std::vector<float>> &probabilities) {
  return probabilities[5][kBoxEnd] + probabilities[5][kNull] +
         probabilities[5][kImEnd];
}

float TopCoordinateProbability(const std::vector<float> &probabilities) {
  return *std::max_element(probabilities.begin() + kCoordStart,
                           probabilities.begin() + kCoordEnd + 1);
}

bool DecodeBox(const std::vector<std::vector<float>> &probabilities,
               std::vector<int32_t> *tokens) {
  const float start = probabilities[0][kBoxStart];
  if (start >= 0.6f && probabilities[1][kNone] > 0.2f &&
      probabilities[2][kBoxEnd] > 0.2f &&
      probabilities[3][kNull] > 0.1f &&
      probabilities[4][kNull] > 0.1f) {
    *tokens = {kBoxStart, kNone, kBoxEnd, kNull, kNull, kNull};
    return true;
  }
  const float end_score = EndScore(probabilities);
  if (end_score < 0.2f) return false;

  std::vector<int32_t> coordinates;
  for (int32_t row = 1; row <= 4; ++row) {
    const auto top = TopK(probabilities[static_cast<size_t>(row)], 4);
    std::vector<int32_t> valid;
    for (int32_t token : top) {
      if (token >= kCoordStart && token <= kCoordEnd) valid.push_back(token);
    }
    if (valid.empty()) return false;
    const int32_t first = valid.front();
    const auto range = std::minmax_element(valid.begin(), valid.end());
    const bool abnormal = probabilities[static_cast<size_t>(row)][first] < 0.9f &&
                          valid.size() > 1 && *range.second - *range.first > 60;
    coordinates.push_back(abnormal ? 0 : first);
  }
  *tokens = {kBoxStart, coordinates[0], coordinates[1], coordinates[2],
             coordinates[3], kBoxEnd};
  return true;
}

bool DecodeRef(const std::vector<std::vector<float>> &probabilities,
               std::vector<int32_t> *tokens) {
  if (probabilities[0][kRefStart] < 0.6f) return false;
  tokens->clear();
  tokens->push_back(kRefStart);
  for (int32_t row = 1; row < 6; ++row) {
    const auto top = TopK(probabilities[static_cast<size_t>(row)], 5);
    auto found = std::find_if(top.begin(), top.end(), [](int32_t token) {
      return token < kCoordStart || token > kCoordEnd;
    });
    if (found == top.end()) return false;
    tokens->push_back(*found);
  }
  return true;
}

HybridDecision HandlePattern(std::vector<int32_t> tokens) {
  if (tokens.empty()) return {"im_end", {kImEnd}, false, true};
  if (tokens[0] == kNull || tokens[0] == kImEnd) {
    return {"im_end", {kImEnd}, false, true};
  }
  if (tokens.size() >= 2 && tokens[0] == kBoxStart && tokens[1] == kNone) {
    return {"empty_box", {kBoxStart, kNone, kBoxEnd}, false, false};
  }
  if (tokens[0] == kBoxStart) {
    int32_t coordinate_count = 1;
    for (size_t index = 1; index < std::min<size_t>(5, tokens.size()); ++index) {
      if (IsCoordinateToken(tokens[index])) ++coordinate_count;
      else break;
    }
    if (coordinate_count == 5 && tokens.size() >= 6 && tokens[5] == kBoxEnd) {
      tokens.resize(6);
      return {"coord_box", std::move(tokens), false, false};
    }
    if (coordinate_count == 3 && tokens.size() >= 4 && tokens[3] == kBoxEnd) {
      tokens.resize(4);
      return {"point_box", std::move(tokens), false, false};
    }
    tokens.resize(static_cast<size_t>(coordinate_count));
    return {"error_box", std::move(tokens), true, false};
  }
  auto null_position = std::find(tokens.begin(), tokens.end(), kNull);
  tokens.erase(null_position, tokens.end());
  if (tokens.size() >= 2 && tokens[tokens.size() - 1] == kRefEnd &&
      tokens[tokens.size() - 2] == kRefEnd) {
    tokens.pop_back();
  }
  return {"ref_object", std::move(tokens), false, false};
}

}  // namespace

bool IsCoordinateToken(int32_t token) {
  return token >= kCoordStart && token <= kCoordEnd;
}

HybridDecision DecodePbd(const Tensor &logits,
                         const std::vector<int32_t> &generated,
                         const PbdDecodeConfig &config,
                         PbdDiagnostics *diagnostics,
                         int32_t row_start) {
  if (logits.dtype != 4 || logits.shape.size() != 3 ||
      logits.shape[0] != 1 || logits.shape[2] != kVocab || row_start < 0 ||
      row_start + 6 > logits.shape[1] ||
      logits.data.size() < static_cast<size_t>(logits.shape[1]) * kVocab * 2) {
    return {"im_end", {kImEnd}, false, true};
  }
  if (config.temperature <= 0.0f || config.top_p <= 0.0f ||
      config.top_p > 1.0f || config.repetition_penalty <= 0.0f) {
    return {"im_end", {kImEnd}, false, true};
  }
  const std::vector<int32_t> &history_tokens = BuildHistoryTokens(generated);
  std::array<PbdRowResult, 6> row_results;
  PbdRows().Decode(logits, history_tokens, config, diagnostics != nullptr,
                   row_start, &row_results);
  std::vector<std::vector<float>> legacy_probabilities;
  if (diagnostics != nullptr) legacy_probabilities.reserve(6);
  std::vector<std::vector<float>> probabilities;
  std::vector<int32_t> greedy;
  for (int32_t row = 0; row < 6; ++row) {
    PbdRowResult &result = row_results[static_cast<size_t>(row)];
    if (diagnostics != nullptr) {
      legacy_probabilities.push_back(std::move(result.legacy_probabilities));
      diagnostics->retained_tokens[static_cast<size_t>(row)] =
          result.retained_tokens;
    }
    greedy.push_back(result.greedy_token);
    probabilities.push_back(std::move(result.probabilities));
  }
  if (diagnostics != nullptr) {
    diagnostics->valid = true;
    diagnostics->legacy_box_start = legacy_probabilities[0][kBoxStart];
    diagnostics->official_box_start = probabilities[0][kBoxStart];
    diagnostics->legacy_ref_start = legacy_probabilities[0][kRefStart];
    diagnostics->official_ref_start = probabilities[0][kRefStart];
    diagnostics->legacy_end_score = EndScore(legacy_probabilities);
    diagnostics->official_end_score = EndScore(probabilities);
    for (int32_t row = 1; row <= 4; ++row) {
      diagnostics->legacy_coord_top[static_cast<size_t>(row - 1)] =
          TopCoordinateProbability(legacy_probabilities[static_cast<size_t>(row)]);
      diagnostics->official_coord_top[static_cast<size_t>(row - 1)] =
          TopCoordinateProbability(probabilities[static_cast<size_t>(row)]);
    }
  }
  std::vector<int32_t> decoded;
  if (!DecodeBox(probabilities, &decoded) &&
      !DecodeRef(probabilities, &decoded)) {
    decoded = std::move(greedy);
  }
  return HandlePattern(std::move(decoded));
}

HybridDecision DecodePbdGreedy(const Tensor &logits,
                               const std::vector<int32_t> &generated) {
  return DecodePbd(logits, generated);
}

HybridDecision DecodePbdCompact(const std::vector<Tensor> &outputs) {
  // Compact BPU sampler contract:
  // sampled ids, global top-5 ids/probabilities, special probabilities,
  // coordinate top-4 ids/probabilities, nucleus mass, then KV updates.
  if (outputs.size() < 7 || outputs[0].dtype != 8 ||
      outputs[0].shape != std::vector<int32_t>{1, 6, 1}) {
    return {"im_end", {kImEnd}, false, true};
  }
  const auto read_i32 = [](const Tensor &tensor, size_t index) -> int32_t {
    if (tensor.dtype != 8 || index * sizeof(int32_t) + sizeof(int32_t) >
            tensor.data.size()) return kImEnd;
    return reinterpret_cast<const int32_t *>(tensor.data.data())[index];
  };
  const auto read_f16 = [](const Tensor &tensor, size_t index) -> float {
    if (tensor.dtype != 4 || index * sizeof(uint16_t) + sizeof(uint16_t) >
            tensor.data.size()) return 0.0f;
    return Fp16ToFloat(reinterpret_cast<const uint16_t *>(tensor.data.data())[index]);
  };
  const Tensor &top_ids = outputs[1];
  const Tensor &top_probs = outputs[2];
  const Tensor &special = outputs[3];
  const Tensor &coord_ids = outputs[4];
  const Tensor &coord_probs = outputs[5];
  if (top_ids.shape != std::vector<int32_t>{1, 6, 5} ||
      top_probs.shape != std::vector<int32_t>{1, 6, 5} ||
      special.shape != std::vector<int32_t>{1, 6, 6} ||
      coord_ids.shape != std::vector<int32_t>{1, 6, 4} ||
      coord_probs.shape != std::vector<int32_t>{1, 6, 4}) {
    return {"im_end", {kImEnd}, false, true};
  }
  auto special_probability = [&](int32_t row, int32_t index) {
    return read_f16(special, static_cast<size_t>(row * 6 + index));
  };
  const float end_score = special_probability(5, 2) +
                          special_probability(5, 3) +
                          special_probability(5, 4);
  std::vector<int32_t> decoded;
  if (special_probability(0, 0) >= 0.6f &&
      special_probability(1, 5) > 0.2f &&
      special_probability(2, 2) > 0.2f &&
      special_probability(3, 3) > 0.1f &&
      special_probability(4, 3) > 0.1f) {
    decoded = {kBoxStart, kNone, kBoxEnd, kNull, kNull, kNull};
  } else if (end_score >= 0.2f) {
    std::vector<int32_t> coordinates;
    for (int32_t row = 1; row <= 4; ++row) {
      std::vector<int32_t> valid;
      for (int32_t rank = 0; rank < 4; ++rank) {
        const size_t index = static_cast<size_t>(row * 4 + rank);
        const int32_t token = read_i32(coord_ids, index);
        if (IsCoordinateToken(token)) valid.push_back(token);
      }
      if (valid.empty()) {
        coordinates.clear();
        break;
      }
      const int32_t first = valid.front();
      const auto range = std::minmax_element(valid.begin(), valid.end());
      const float first_probability = read_f16(
          coord_probs, static_cast<size_t>(row * 4));
      const bool abnormal = first_probability < 0.9f && valid.size() > 1 &&
                            *range.second - *range.first > 60;
      coordinates.push_back(abnormal ? 0 : first);
    }
    if (coordinates.size() == 4) {
      decoded = {kBoxStart, coordinates[0], coordinates[1], coordinates[2],
                 coordinates[3], kBoxEnd};
    }
  }
  if (decoded.empty() && special_probability(0, 1) >= 0.6f) {
    decoded.push_back(kRefStart);
    for (int32_t row = 1; row < 6; ++row) {
      int32_t selected = kImEnd;
      for (int32_t rank = 0; rank < 5; ++rank) {
        const int32_t token = read_i32(top_ids, static_cast<size_t>(row * 5 + rank));
        if (token < kCoordStart || token > kCoordEnd) {
          selected = token;
          break;
        }
      }
      if (selected == kImEnd) {
        decoded.clear();
        break;
      }
      decoded.push_back(selected);
    }
  }
  if (decoded.empty()) {
    for (int32_t row = 0; row < 6; ++row) {
      decoded.push_back(read_i32(outputs[0], static_cast<size_t>(row)));
    }
  }
  return HandlePattern(std::move(decoded));
}

int32_t DecodeArGreedy(const Tensor &logits,
                       const std::vector<int32_t> &generated) {
  if (logits.dtype != 4 || logits.shape != std::vector<int32_t>{1, 1, kVocab}) {
    return kImEnd;
  }
  const std::vector<int32_t> &history_tokens = BuildHistoryTokens(generated);
  return Argmax(Row(logits, 0, history_tokens, 1.1f));
}

std::string RenderLocateAnythingTokens(const std::vector<int32_t> &tokens) {
  std::ostringstream output;
  for (int32_t token : tokens) {
    if (token == kBoxStart) output << "<box>";
    else if (token == kBoxEnd) output << "</box>";
    else if (token == kRefStart) output << "<ref>";
    else if (token == kRefEnd) output << "</ref>";
    else if (token == kNull) output << "<null>";
    else if (token == kImEnd) output << "<|im_end|>";
    else if (token == kNone) output << "none";
    else if (token == kTextMask) output << "<text_mask>";
    else if (IsCoordinateToken(token)) output << '<' << token - kCoordStart << '>';
    else output << "<tok:" << token << ">";
  }
  return output.str();
}

}  // namespace locateanything_runtime

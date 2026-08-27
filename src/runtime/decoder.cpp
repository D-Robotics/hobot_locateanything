#include "runtime/decoder.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <condition_variable>
#include <mutex>
#include <numeric>
#include <limits>
#include <sstream>
#include <stdexcept>
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

/**
 * @brief Convert a finite IEEE-754 binary16 value to a host float.
 * @param bits Raw fp16 value.
 * @return Converted fp32 value; infinities and NaNs are normalized to NaN so
 * they cannot participate in host probability ranking.
 */
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

/**
 * @brief Convert one vocabulary row from fp16 logits to host-side fp32.
 * @param raw Source row containing exactly one vocabulary of fp16 values.
 * @param values Pre-sized fp32 destination vector.
 */
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

/**
 * @brief Extract and repetition-adjust one logits row.
 * @param logits Source fp16 logits tensor.
 * @param row Sequence-axis row index.
 * @param history_tokens De-duplicated generated-token history.
 * @param repetition_penalty Positive repetition penalty.
 * @param values Reusable fp32 vocabulary destination.
 */
void FillRow(const Tensor &logits, int32_t row,
             const std::vector<int32_t> &history_tokens,
             float repetition_penalty, std::vector<float> *values) {
  const auto *raw = reinterpret_cast<const uint16_t *>(logits.data.data());
  values->resize(kVocab);
  const size_t offset = static_cast<size_t>(row) * kVocab;
  ConvertFp16Row(raw + offset, values);
  if (repetition_penalty != 1.0f) {
    for (const int32_t token : history_tokens) {
      float &value = (*values)[static_cast<size_t>(token)];
      value = value > 0 ? value / repetition_penalty
                        : value * repetition_penalty;
    }
  }
}

struct RowWorkspace {
  /**
   * @brief Allocate reusable vocabulary-sized score and index buffers.
   */
  RowWorkspace() : values(kVocab), indices(kVocab) {}

  std::vector<float> values;
  std::vector<int32_t> indices;
};

/**
 * @brief Return the first index containing the largest vector value.
 * @param[in] values Candidate scores.
 * @return Zero-based index of the maximum value, preserving the first tie.
 */
int32_t Argmax(const std::vector<float> &values);

/**
 * @brief Apply temperature scaling and top-p filtering to one logits row.
 * @param values Repetition-adjusted vocabulary scores, replaced by filtered
 * probabilities.
 * @param indices Reusable sorting workspace.
 * @param temperature Positive sampling temperature.
 * @param top_p Retained probability mass in (0, 1].
 * @return Retained-token count.
 */
int32_t NucleusSoftmax(std::vector<float> *values,
                       std::vector<int32_t> *indices,
                       float temperature, float top_p) {
  const float inverse_temperature = 1.0f / temperature;
  float maximum = -std::numeric_limits<float>::infinity();
  float second_maximum = -std::numeric_limits<float>::infinity();
  size_t maximum_token = 0;
  for (size_t index = 0; index < values->size(); ++index) {
    const float value = (*values)[index] * inverse_temperature;
    (*values)[index] = value;
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
        static_cast<double>(values->size() - 1) *
        std::exp(static_cast<double>(second_maximum - maximum));
    const double maximum_probability_lower_bound =
        1.0 / (1.0 + other_mass_upper_bound);
    if (maximum_probability_lower_bound >
        static_cast<double>(top_p) + 1e-6) {
      std::fill(values->begin(), values->end(), 0.0f);
      (*values)[maximum_token] = 1.0f;
      return 1;
    }
  }

  double total = 0.0;
  for (float &value : *values) {
    value = std::exp(value - maximum);
    total += value;
  }
  if (total > 0.0 && std::isfinite(total)) {
    for (float &value : *values) value = static_cast<float>(value / total);
  }
  if (top_p >= 1.0f) {
    return static_cast<int32_t>(values->size());
  }

  const auto maximum_probability =
      std::max_element(values->begin(), values->end());
  if (maximum_probability != values->end() &&
      *maximum_probability > top_p) {
    const size_t token =
        static_cast<size_t>(maximum_probability - values->begin());
    std::fill(values->begin(), values->end(), 0.0f);
    (*values)[token] = 1.0f;
    return 1;
  }

  indices->resize(values->size());
  std::iota(indices->begin(), indices->end(), 0);
  auto greater = [&](int32_t left, int32_t right) {
    const float lhs = (*values)[static_cast<size_t>(left)];
    const float rhs = (*values)[static_cast<size_t>(right)];
    return lhs > rhs || (lhs == rhs && left < right);
  };
  double cumulative = 0.0;
  size_t retained = 0;
  size_t width = std::min<size_t>(kNucleusInitialWidth, indices->size());
  size_t partitioned = 0;
  while (true) {
    if (width < indices->size()) {
      std::nth_element(indices->begin() + partitioned,
                       indices->begin() + width, indices->end(), greater);
      std::sort(indices->begin(), indices->begin() + width, greater);
    } else {
      std::sort(indices->begin() + partitioned, indices->end(), greater);
    }
    cumulative = 0.0;
    retained = 0;
    for (; retained < width; ++retained) {
      cumulative += (*values)[static_cast<size_t>((*indices)[retained])];
      if (cumulative >= top_p) {
        ++retained;
        break;
      }
    }
    if (cumulative >= top_p || width == indices->size()) break;
    partitioned = width;
    if (width >= static_cast<size_t>(kNucleusMaximumWidth)) {
      width = indices->size();
    } else {
      width = std::min({width * 2, indices->size(),
                        static_cast<size_t>(kNucleusMaximumWidth)});
    }
  }

  if (retained == 0 || cumulative <= 0.0 || !std::isfinite(cumulative)) {
    std::fill(values->begin(), values->end(), 0.0f);
    return 0;
  }
  for (size_t index = retained; index < indices->size(); ++index) {
    (*values)[static_cast<size_t>((*indices)[index])] = 0.0f;
  }
  for (size_t index = 0; index < retained; ++index) {
    const size_t token = static_cast<size_t>((*indices)[index]);
    (*values)[token] = static_cast<float>((*values)[token] / cumulative);
  }
  return static_cast<int32_t>(retained);
}

struct PbdRowResult {
  std::array<int32_t, 5> top_tokens{};
  std::array<float, 5> top_probabilities{};
  int32_t greedy_token = 0;
  float box_start = 0.0f;
  float ref_start = 0.0f;
  float box_end = 0.0f;
  float null_token = 0.0f;
  float image_end = 0.0f;
  float none = 0.0f;
  float top_coordinate = 0.0f;
};

/**
 * @brief Extract greedy, top-k, coordinate, and special-token probabilities.
 * @param[in] probabilities One normalized vocabulary distribution.
 * @param[out] result Reusable row summary consumed by pattern matching.
 */
void SummarizeProbabilities(const std::vector<float> &probabilities,
                            PbdRowResult *result) {
  int32_t top_count = 0;
  float top_coordinate = -std::numeric_limits<float>::infinity();
  auto better = [&](int32_t left, int32_t right) {
    const float lhs = probabilities[static_cast<size_t>(left)];
    const float rhs = probabilities[static_cast<size_t>(right)];
    return lhs > rhs || (lhs == rhs && left < right);
  };
  for (int32_t token = 0; token < kVocab; ++token) {
    if (token >= kCoordStart && token <= kCoordEnd) {
      top_coordinate =
          std::max(top_coordinate, probabilities[static_cast<size_t>(token)]);
    }
    size_t position = 0;
    if (top_count < static_cast<int32_t>(result->top_tokens.size())) {
      position = static_cast<size_t>(top_count++);
      result->top_tokens[position] = token;
    } else if (!better(token, result->top_tokens.back())) {
      continue;
    } else {
      position = result->top_tokens.size() - 1;
      result->top_tokens[position] = token;
    }
    while (position > 0 &&
           better(result->top_tokens[position],
                  result->top_tokens[position - 1])) {
      std::swap(result->top_tokens[position],
                result->top_tokens[position - 1]);
      --position;
    }
  }
  for (size_t rank = 0; rank < result->top_tokens.size(); ++rank) {
    result->top_probabilities[rank] =
        probabilities[static_cast<size_t>(result->top_tokens[rank])];
  }
  result->greedy_token = result->top_tokens[0];
  result->box_start = probabilities[kBoxStart];
  result->ref_start = probabilities[kRefStart];
  result->box_end = probabilities[kBoxEnd];
  result->null_token = probabilities[kNull];
  result->image_end = probabilities[kImEnd];
  result->none = probabilities[kNone];
  result->top_coordinate = top_coordinate;
}

/**
 * @brief Decode one PBD row into the probabilities used by pattern matching.
 * @param logits Source PBD logits tensor.
 * @param row Sequence-axis row index.
 * @param history_tokens De-duplicated generated-token history.
 * @param config Sampling configuration.
 * @param workspace Reusable vocabulary conversion and sorting storage.
 * @return Probability summaries, ranked tokens, and the greedy token for one
 * row after repetition adjustment and nucleus filtering.
 */
PbdRowResult DecodePbdRow(const Tensor &logits, int32_t row,
                          const std::vector<int32_t> &history_tokens,
                          const PbdDecodeConfig &config,
                          RowWorkspace *workspace) {
  FillRow(logits, row, history_tokens, config.repetition_penalty,
          &workspace->values);
  PbdRowResult result;
  (void)NucleusSoftmax(&workspace->values, &workspace->indices,
                       config.temperature, config.top_p);
  SummarizeProbabilities(workspace->values, &result);
  return result;
}

class PbdRowExecutor {
 public:
  /**
   * @brief Start five workers; the caller decodes row six on its own thread.
   */
  PbdRowExecutor() {
    workers_.reserve(5);
    for (int32_t row = 0; row < 5; ++row) {
      workers_.emplace_back([this, row] { Worker(row); });
    }
  }

  /**
   * @brief Stop workers and wait for any in-flight row jobs to finish.
   */
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

  /**
   * @brief Dispatch five rows and synchronously collect all six row results.
   * @param logits Source PBD logits tensor.
   * @param history_tokens De-duplicated generated-token history.
   * @param config Sampling configuration.
   * @param row_start First row of the six-row decision window.
   * @param results Destination fixed-size row result array.
   */
  void Decode(const Tensor &logits,
              const std::vector<int32_t> &history_tokens,
              const PbdDecodeConfig &config, int32_t row_start,
              std::array<PbdRowResult, 6> *results) {
    std::lock_guard<std::mutex> call_lock(call_mutex_);
    {
      std::lock_guard<std::mutex> lock(job_mutex_);
      logits_ = &logits;
      history_tokens_ = &history_tokens;
      config_ = config;
      row_start_ = row_start;
      results_ = results;
      pending_workers_ = 5;
      ++generation_;
    }
    job_ready_.notify_all();

    (*results)[5] =
        DecodePbdRow(logits, row_start + 5, history_tokens, config,
                     &workspaces_[5]);

    std::unique_lock<std::mutex> lock(job_mutex_);
    job_done_.wait(lock, [this] { return pending_workers_ == 0; });
    logits_ = nullptr;
    history_tokens_ = nullptr;
    results_ = nullptr;
  }

 private:
  /**
   * @brief Wait for jobs, decode one assigned row, and signal completion.
   * @param row Worker-owned row offset from zero through four.
   */
  void Worker(int32_t row) {
    size_t observed_generation = 0;
    while (true) {
      const Tensor *logits = nullptr;
      const std::vector<int32_t> *history_tokens = nullptr;
      PbdDecodeConfig config;
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
        row_start = row_start_;
        results = results_;
      }

      (*results)[static_cast<size_t>(row)] =
          DecodePbdRow(*logits, row_start + row, *history_tokens, config,
                       &workspaces_[static_cast<size_t>(row)]);

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
  std::array<RowWorkspace, 6> workspaces_;
  const Tensor *logits_ = nullptr;
  const std::vector<int32_t> *history_tokens_ = nullptr;
  PbdDecodeConfig config_;
  std::array<PbdRowResult, 6> *results_ = nullptr;
  size_t generation_ = 0;
  int32_t pending_workers_ = 0;
  int32_t row_start_ = 0;
  bool stopping_ = false;
};

/**
 * @brief Return the process-wide row executor used by the inference core.
 * @return Long-lived executor with five fixed worker threads.
 */
PbdRowExecutor &PbdRows() {
  static PbdRowExecutor executor;
  return executor;
}

/**
 * @brief Build a de-duplicated token history for repetition penalties.
 * @param generated Prompt and response token history.
 * @return Thread-local valid token IDs, unique in first-occurrence order.
 */
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

/**
 * @brief Return the first index with the largest score.
 * @param values Scores indexed by token ID.
 * @return Index of the maximum value, or zero for an empty vector.
 */
int32_t Argmax(const std::vector<float> &values) {
  return static_cast<int32_t>(
      std::max_element(values.begin(), values.end()) - values.begin());
}

/**
 * @brief Recognize a complete box or explicit empty-box pattern.
 * @param[in] rows Six summarized PBD vocabulary distributions.
 * @param[out] tokens Destination accepted box tokens.
 * @return True when a valid complete pattern was recognized.
 */
bool DecodeBox(const std::array<PbdRowResult, 6> &rows,
               std::vector<int32_t> *tokens) {
  const float start = rows[0].box_start;
  if (start >= 0.6f && rows[1].none > 0.2f &&
      rows[2].box_end > 0.2f && rows[3].null_token > 0.1f &&
      rows[4].null_token > 0.1f) {
    *tokens = {kBoxStart, kNone, kBoxEnd, kNull, kNull, kNull};
    return true;
  }
  const float end_score =
      rows[5].box_end + rows[5].null_token + rows[5].image_end;
  if (end_score < 0.2f) return false;

  std::vector<int32_t> coordinates;
  for (int32_t row = 1; row <= 4; ++row) {
    const PbdRowResult &result = rows[static_cast<size_t>(row)];
    std::vector<int32_t> valid;
    float first_probability = 0.0f;
    for (size_t rank = 0; rank < 4; ++rank) {
      const int32_t token = result.top_tokens[rank];
      if (token >= kCoordStart && token <= kCoordEnd) {
        if (valid.empty()) first_probability = result.top_probabilities[rank];
        valid.push_back(token);
      }
    }
    if (valid.empty()) return false;
    const int32_t first = valid.front();
    const auto range = std::minmax_element(valid.begin(), valid.end());
    const bool abnormal = first_probability < 0.9f && valid.size() > 1 &&
                          *range.second - *range.first > 60;
    coordinates.push_back(abnormal ? 0 : first);
  }
  *tokens = {kBoxStart, coordinates[0], coordinates[1], coordinates[2],
             coordinates[3], kBoxEnd};
  return true;
}

/**
 * @brief Recognize a six-row reference-object token pattern.
 * @param[in] rows Six summarized PBD vocabulary distributions.
 * @param[out] tokens Destination accepted reference tokens.
 * @return True when a valid reference pattern was recognized.
 */
bool DecodeRef(const std::array<PbdRowResult, 6> &rows,
               std::vector<int32_t> *tokens) {
  if (rows[0].ref_start < 0.6f) return false;
  tokens->clear();
  tokens->push_back(kRefStart);
  for (int32_t row = 1; row < 6; ++row) {
    const auto &top = rows[static_cast<size_t>(row)].top_tokens;
    auto found = std::find_if(top.begin(), top.end(), [](int32_t token) {
      return token < kCoordStart || token > kCoordEnd;
    });
    if (found == top.end()) return false;
    tokens->push_back(*found);
  }
  return true;
}

/**
 * @brief Convert a candidate token pattern into a hybrid decoder decision.
 * @param tokens Candidate six-row token sequence.
 * @return Accepted tokens and PBD/AR/terminal control flags.
 */
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

/**
 * @brief Check whether a token represents a LocateAnything coordinate.
 * @param[in] token Model token ID.
 * @return True for coordinate values 0 through 1000; otherwise false.
 */
bool IsCoordinateToken(int32_t token) {
  return token >= kCoordStart && token <= kCoordEnd;
}

/**
 * @brief Decode one six-row PBD logits window and select the next decode step.
 * @param[in] logits FP16 logits with shape [1, rows, vocabulary].
 * @param[in] generated Prompt and accepted response history used by repetition
 *                      control and structured-token handling.
 * @param[in] config Temperature, nucleus-sampling, and repetition settings.
 * @param[in] row_start Index of the first of six rows to decode.
 * @return Accepted token IDs plus terminal, PBD-continuation, or AR-fallback
 *         control state.
 * @throws std::invalid_argument if the tensor, row window, or sampling
 *         configuration is invalid.
 */
HybridDecision DecodePbd(const Tensor &logits,
                         const std::vector<int32_t> &generated,
                         const PbdDecodeConfig &config,
                         int32_t row_start) {
  if (logits.dtype != 4 || logits.shape.size() != 3 ||
      logits.shape[0] != 1 || logits.shape[2] != kVocab || row_start < 0 ||
      row_start + 6 > logits.shape[1] ||
      logits.data.size() < static_cast<size_t>(logits.shape[1]) * kVocab * 2) {
    throw std::invalid_argument("PBD logits must contain six valid FP16 rows");
  }
  if (config.temperature <= 0.0f || config.top_p <= 0.0f ||
      config.top_p > 1.0f || config.repetition_penalty <= 0.0f) {
    throw std::invalid_argument("PBD sampling configuration is invalid");
  }
  const std::vector<int32_t> &history_tokens = BuildHistoryTokens(generated);
  std::array<PbdRowResult, 6> row_results;
  PbdRows().Decode(logits, history_tokens, config, row_start, &row_results);
  std::vector<int32_t> greedy;
  greedy.reserve(6);
  for (int32_t row = 0; row < 6; ++row) {
    PbdRowResult &result = row_results[static_cast<size_t>(row)];
    greedy.push_back(result.greedy_token);
  }
  std::vector<int32_t> decoded;
  if (!DecodeBox(row_results, &decoded) && !DecodeRef(row_results, &decoded)) {
    decoded = std::move(greedy);
  }
  return HandlePattern(std::move(decoded));
}

/**
 * @brief Decode a PBD window with the default LocateAnything sampling policy.
 * @param[in] logits FP16 logits with shape [1, rows, vocabulary].
 * @param[in] generated Prompt and accepted response token history.
 * @return Accepted tokens and the next hybrid-decoder control decision.
 * @throws std::invalid_argument if the logits tensor is incompatible.
 */
HybridDecision DecodePbdGreedy(const Tensor &logits,
                               const std::vector<int32_t> &generated) {
  return DecodePbd(logits, generated);
}

/**
 * @brief Select one autoregressive token from the final logits row.
 * @param[in] logits FP16 logits with shape [1, 1, vocabulary].
 * @param[in] generated Prompt and response history used for repetition penalty.
 * @return Selected vocabulary token ID.
 * @throws std::invalid_argument if logits do not contain one valid FP16 row.
 */
int32_t DecodeArGreedy(const Tensor &logits,
                       const std::vector<int32_t> &generated) {
  if (logits.dtype != 4 || logits.shape != std::vector<int32_t>{1, 1, kVocab}) {
    throw std::invalid_argument("AR logits must be one FP16 vocabulary row");
  }
  const std::vector<int32_t> &history_tokens = BuildHistoryTokens(generated);
  static thread_local std::vector<float> values(kVocab);
  FillRow(logits, 0, history_tokens, 1.1f, &values);
  return Argmax(values);
}

/**
 * @brief Render generated token IDs as LocateAnything diagnostic markup.
 * @param[in] tokens Generated response token IDs.
 * @return Human-readable text containing known structural and coordinate
 *         tokens; unknown IDs are rendered numerically.
 */
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

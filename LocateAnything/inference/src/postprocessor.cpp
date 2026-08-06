#include "postprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "tokenizer.hpp"

namespace locateanything {
namespace {

constexpr float kCanvasSize = 672.0f;

float RestoreCoordinate(int32_t value, bool vertical,
                        const ImageTransform& transform) {
  const float padding = vertical ? transform.pad_top : transform.pad_left;
  const float scale = vertical ? transform.scale_y : transform.scale_x;
  const float limit = static_cast<float>(vertical ? transform.source_height
                                                   : transform.source_width);
  const float pixel = (static_cast<float>(value) / 1000.0f * kCanvasSize - padding) /
                      scale;
  return std::clamp(pixel, 0.0f, limit);
}

float IoU(const Detection& left, const Detection& right) {
  const float x1 = std::max(left.box[0], right.box[0]);
  const float y1 = std::max(left.box[1], right.box[1]);
  const float x2 = std::min(left.box[2], right.box[2]);
  const float y2 = std::min(left.box[3], right.box[3]);
  const float intersection = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
  const float left_area = std::max(0.0f, left.box[2] - left.box[0]) *
                          std::max(0.0f, left.box[3] - left.box[1]);
  const float right_area = std::max(0.0f, right.box[2] - right.box[0]) *
                           std::max(0.0f, right.box[3] - right.box[1]);
  const float denominator = left_area + right_area - intersection;
  return denominator > 0.0f ? intersection / denominator : 0.0f;
}

std::string CanonicalLabel(std::string value) {
  std::string result;
  bool pending_space = false;
  for (unsigned char item : value) {
    if (std::isspace(item)) {
      pending_space = !result.empty();
      continue;
    }
    if (pending_space) result.push_back(' ');
    pending_space = false;
    result.push_back(static_cast<char>(std::tolower(item)));
  }
  return result;
}

std::string JsonEscape(const std::string& value) {
  std::ostringstream output;
  for (unsigned char item : value) {
    switch (item) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (item < 0x20U) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<int>(item) << std::dec;
        } else {
          output << static_cast<char>(item);
        }
    }
  }
  return output.str();
}

}  // namespace

Postprocessor::Postprocessor(float nms_iou) : nms_iou_(nms_iou) {
  if (nms_iou < 0.0f || nms_iou > 1.0f) {
    throw std::invalid_argument("NMS IoU must be between zero and one");
  }
}

Prediction Postprocessor::Parse(const std::vector<int32_t>& tokens,
                                const ImageTransform& transform,
                                const Tokenizer& tokenizer,
                                const std::string& task) const {
  const int32_t ref_start = tokenizer.TokenId("<ref>");
  const int32_t ref_end = tokenizer.TokenId("</ref>");
  const int32_t box_start = tokenizer.TokenId("<box>");
  const int32_t box_end = tokenizer.TokenId("</box>");
  const int32_t coordinate_start = tokenizer.TokenId("<0>");
  const int32_t coordinate_end = tokenizer.TokenId("<1000>");
  if (ref_start < 0 || ref_end < 0 || box_start < 0 || box_end < 0 ||
      coordinate_start < 0 || coordinate_end < coordinate_start) {
    throw std::runtime_error("tokenizer is missing LocateAnything output tokens");
  }

  Prediction result;
  std::string label;
  for (size_t index = 0; index < tokens.size();) {
    if (tokens[index] == ref_start) {
      const auto end = std::find(tokens.begin() + static_cast<std::ptrdiff_t>(index + 1),
                                 tokens.end(), ref_end);
      if (end == tokens.end()) break;
      label = tokenizer.Decode(std::vector<int32_t>(
          tokens.begin() + static_cast<std::ptrdiff_t>(index + 1), end));
      index = static_cast<size_t>(std::distance(tokens.begin(), end)) + 1;
      continue;
    }
    if (tokens[index] != box_start) {
      ++index;
      continue;
    }
    const auto end = std::find(tokens.begin() + static_cast<std::ptrdiff_t>(index + 1),
                               tokens.end(), box_end);
    if (end == tokens.end()) break;
    std::vector<int32_t> coordinates;
    for (auto item = tokens.begin() + static_cast<std::ptrdiff_t>(index + 1);
         item != end; ++item) {
      if (*item < coordinate_start || *item > coordinate_end) {
        coordinates.clear();
        break;
      }
      coordinates.push_back(*item - coordinate_start);
    }
    if (coordinates.size() == 4 && coordinates[0] < coordinates[2] &&
        coordinates[1] < coordinates[3]) {
      Detection detection;
      detection.label = label;
      std::copy(coordinates.begin(), coordinates.end(),
                detection.normalized_box.begin());
      for (size_t axis = 0; axis < 4; ++axis) {
        detection.box[axis] = RestoreCoordinate(coordinates[axis], axis % 2 == 1,
                                                transform);
      }
      if (detection.box[0] < detection.box[2] && detection.box[1] < detection.box[3]) {
        result.detections.push_back(std::move(detection));
      }
    } else if (coordinates.size() == 2) {
      Point point;
      point.label = label;
      std::copy(coordinates.begin(), coordinates.end(),
                point.normalized_point.begin());
      point.point = {RestoreCoordinate(coordinates[0], false, transform),
                     RestoreCoordinate(coordinates[1], true, transform)};
      result.points.push_back(std::move(point));
    }
    index = static_cast<size_t>(std::distance(tokens.begin(), end)) + 1;
  }

  if (task == "object_detection") {
    std::vector<Detection> kept;
    for (const Detection& detection : result.detections) {
      const std::string label_key = CanonicalLabel(detection.label);
      const bool duplicate = std::any_of(kept.begin(), kept.end(), [&](const Detection& item) {
        return CanonicalLabel(item.label) == label_key && IoU(item, detection) >= nms_iou_;
      });
      if (!duplicate) kept.push_back(detection);
    }
    result.detections = std::move(kept);
  }
  return result;
}

cv::Mat Postprocessor::Draw(const cv::Mat& source,
                            const Prediction& prediction) const {
  cv::Mat image = source.clone();
  const std::array<cv::Scalar, 6> colors = {
      cv::Scalar(120, 220, 0), cv::Scalar(255, 145, 0),
      cv::Scalar(75, 90, 255), cv::Scalar(0, 190, 255),
      cv::Scalar(255, 95, 175), cv::Scalar(205, 205, 0)};
  const int line_width = std::max(2, static_cast<int>(
                                         std::lround(std::min(image.cols, image.rows) * 0.004)));
  size_t index = 0;
  for (const Detection& detection : prediction.detections) {
    const cv::Scalar color = colors[index % colors.size()];
    const cv::Rect rectangle(
        cv::Point(static_cast<int>(std::lround(detection.box[0])),
                  static_cast<int>(std::lround(detection.box[1]))),
        cv::Point(static_cast<int>(std::lround(detection.box[2])),
                  static_cast<int>(std::lround(detection.box[3]))));
    cv::rectangle(image, rectangle, color, line_width);
    const std::string caption = std::to_string(index + 1) +
                                (detection.label.empty() ? "" : ": " + detection.label);
    cv::putText(image, caption,
                cv::Point(std::max(0, rectangle.x), std::max(18, rectangle.y + 18)),
                cv::FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv::LINE_AA);
    ++index;
  }
  for (const Point& point : prediction.points) {
    const cv::Scalar color = colors[index % colors.size()];
    const cv::Point center(static_cast<int>(std::lround(point.point[0])),
                           static_cast<int>(std::lround(point.point[1])));
    cv::circle(image, center, std::max(5, line_width * 2), color, line_width,
               cv::LINE_AA);
    ++index;
  }
  return image;
}

std::string Postprocessor::ToJson(const Prediction& prediction,
                                  const std::string& task,
                                  const std::string& stop_reason,
                                  uint64_t frame_index,
                                  const InferenceMetrics& metrics) const {
  std::ostringstream output;
  output << std::fixed << std::setprecision(3)
         << "{\"frame_index\":" << frame_index << ",\"task\":\""
         << JsonEscape(task) << "\",\"stop_reason\":\""
         << JsonEscape(stop_reason) << "\",\"detections\":[";
  for (size_t index = 0; index < prediction.detections.size(); ++index) {
    if (index != 0) output << ',';
    const Detection& item = prediction.detections[index];
    output << "{\"label\":\"" << JsonEscape(item.label) << "\",\"bbox_xyxy\":["
           << item.box[0] << ',' << item.box[1] << ',' << item.box[2] << ','
           << item.box[3] << "]}";
  }
  output << "],\"points\":[";
  for (size_t index = 0; index < prediction.points.size(); ++index) {
    if (index != 0) output << ',';
    const Point& item = prediction.points[index];
    output << "{\"label\":\"" << JsonEscape(item.label) << "\",\"point_xy\":["
           << item.point[0] << ',' << item.point[1] << "]}";
  }
  const LanguageMetrics& language = metrics.language;
  const double tokens_per_second =
      language.decode_ms > 0.0
          ? static_cast<double>(language.generated_tokens) * 1000.0 /
                language.decode_ms
          : 0.0;
  output << "],\"generation\":{\"prompt_tokens\":" << language.prompt_tokens
          << ",\"generated_tokens\":" << language.generated_tokens
          << ",\"tokens_per_second\":" << tokens_per_second
         << ",\"executed_mode\":\"" << JsonEscape(language.executed_mode)
         << "\",\"fallback_reason\":\"" << JsonEscape(language.fallback_reason)
         << "\",\"pbd_calls\":" << language.pbd_calls
         << ",\"pbd_accepted_tokens\":" << language.pbd_accepted_tokens
         << ",\"ar_calls\":" << language.ar_calls
         << ",\"ar_tokens\":" << language.ar_tokens
         << ",\"graph_timings\":[";
  for (size_t index = 0; index < language.graph_timings.size(); ++index) {
    if (index != 0) output << ',';
    const GraphTiming& timing = language.graph_timings[index];
    output << "{\"graph\":\"" << JsonEscape(timing.graph)
           << "\",\"calls\":" << timing.calls
           << ",\"total_ms\":" << timing.total_ms
           << ",\"bpu_wait_ms\":" << timing.bpu_wait_ms
           << ",\"submit_ms\":" << timing.submit_ms
           << ",\"input_bytes\":" << timing.input_bytes
           << ",\"output_bytes\":" << timing.output_bytes << '}';
  }
  output << "],\"cache_update_ms\":" << language.cache_update_ms
         << ",\"host_decode_ms\":" << language.host_decode_ms
         << "},\"timing_ms\":{\"preprocess\":" << metrics.preprocess_ms
         << ",\"vision\":" << metrics.vision_ms
         << ",\"language_prefill\":" << language.prefill_ms
         << ",\"language_decode\":" << language.decode_ms
         << ",\"language\":" << metrics.language_ms
         << ",\"total\":" << metrics.total_ms << "}}";
  return output.str();
}

}  // namespace locateanything

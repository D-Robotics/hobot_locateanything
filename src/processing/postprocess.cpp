#include "processing/postprocess.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

#include "processing/tokenizer.hpp"

namespace locateanything {
namespace {

const std::array<cv::Scalar, 8> kAnnotationColors = {
    cv::Scalar(255, 184, 45), cv::Scalar(66, 214, 164),
    cv::Scalar(80, 126, 255), cv::Scalar(72, 202, 255),
    cv::Scalar(224, 112, 255), cv::Scalar(255, 120, 128),
    cv::Scalar(128, 224, 96), cv::Scalar(255, 170, 96)};

/**
 * @brief Paint a translucent rectangle without changing pixels outside it.
 * @param image Destination BGR image.
 * @param rectangle Region to tint.
 * @param color Overlay color in BGR order.
 * @param opacity Overlay opacity in [0, 1].
 */
void FillTranslucent(cv::Mat* image, const cv::Rect& rectangle,
                     const cv::Scalar& color, double opacity) {
  const cv::Rect bounds(0, 0, image->cols, image->rows);
  const cv::Rect clipped = rectangle & bounds;
  if (clipped.empty()) return;
  cv::Mat region = (*image)(clipped);
  cv::Mat overlay(region.size(), region.type(), color);
  cv::addWeighted(overlay, opacity, region, 1.0 - opacity, 0.0, region);
}

/**
 * @brief Draw a readable caption next to an annotation anchor.
 * @param image Destination BGR image.
 * @param caption Text rendered with OpenCV's ASCII font.
 * @param anchor Top-left target coordinate.
 * @param color Accent color assigned to the target.
 * @param font_scale Font scale derived from source resolution.
 * @param text_width Text stroke width.
 * @param prefer_above Place the caption above the anchor when space permits.
 * @return Final caption rectangle in image coordinates.
 */
cv::Rect DrawCaption(cv::Mat* image, const std::string& caption,
                     const cv::Point& anchor, const cv::Scalar& color,
                     double font_scale, int text_width, bool prefer_above) {
  int baseline = 0;
  const cv::Size text = cv::getTextSize(caption, cv::FONT_HERSHEY_SIMPLEX,
                                        font_scale, text_width, &baseline);
  const int padding_x = std::max(6, text_width * 3);
  const int padding_y = std::max(4, text_width * 2);
  const int accent_width = std::max(4, text_width * 2);
  const int width = text.width + padding_x * 2 + accent_width;
  const int height = text.height + baseline + padding_y * 2;

  int x = std::clamp(anchor.x, 0, std::max(0, image->cols - width));
  int y = anchor.y;
  if (prefer_above && anchor.y >= height + 2) y = anchor.y - height - 2;
  y = std::clamp(y, 0, std::max(0, image->rows - height));
  const cv::Rect background(x, y, std::min(width, image->cols - x),
                            std::min(height, image->rows - y));
  FillTranslucent(image, background, cv::Scalar(18, 22, 28), 0.90);
  cv::rectangle(*image,
                cv::Rect(background.x, background.y,
                         std::min(accent_width, background.width),
                         background.height),
                color, cv::FILLED, cv::LINE_AA);
  const cv::Point origin(background.x + accent_width + padding_x,
                         background.y + padding_y + text.height);
  cv::putText(*image, caption, origin, cv::FONT_HERSHEY_SIMPLEX, font_scale,
              cv::Scalar(250, 250, 250), text_width, cv::LINE_AA);
  return background;
}

/**
 * @brief Map one normalized model coordinate back to a source pixel.
 * @param value Model coordinate in the 0..1000 range.
 * @param vertical Select the vertical transform when true.
 * @param transform Resize and padding metadata.
 * @return Clamped source-image coordinate.
 */
float RestoreCoordinate(int32_t value, bool vertical,
                        const ImageTransform& transform) {
  const float padding = vertical ? transform.pad_top : transform.pad_left;
  const float scale = vertical ? transform.scale_y : transform.scale_x;
  const float canvas = static_cast<float>(
      vertical ? transform.canvas_height : transform.canvas_width);
  const float limit = static_cast<float>(vertical ? transform.source_height
                                                   : transform.source_width);
  if (canvas <= 0.0f || scale <= 0.0f) {
    throw std::invalid_argument("invalid image transform");
  }
  const float pixel = (static_cast<float>(value) / 1000.0f * canvas - padding) /
                      scale;
  return std::clamp(pixel, 0.0f, limit);
}

/**
 * @brief Compute intersection-over-union for two source-space boxes.
 * @param left First detection.
 * @param right Second detection.
 * @return IoU in [0, 1].
 */
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

/**
 * @brief Normalize a label for case-insensitive duplicate comparison.
 * @param value Raw decoded label.
 * @return Lowercase label with normalized whitespace.
 */
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

/**
 * @brief Escape control characters and quotes for a JSON string field.
 * @param value Raw UTF-8 field value.
 * @return JSON-escaped field contents without surrounding quotes.
 */
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
  if (image.empty()) return image;
  const int short_edge = std::min(image.cols, image.rows);
  const int line_width = std::clamp(
      static_cast<int>(std::lround(short_edge * 0.0045)), 2, 6);
  const int text_width = std::clamp(line_width - 1, 1, 3);
  const double font_scale = std::clamp(short_edge / 900.0, 0.52, 0.88);
  const size_t target_count = prediction.detections.size() + prediction.points.size();
  std::string shared_label;
  bool compact_labels = target_count > 1;
  const auto inspect_label = [&](const std::string& label) {
    const std::string canonical = CanonicalLabel(label);
    if (canonical.empty()) {
      compact_labels = false;
    } else if (shared_label.empty()) {
      shared_label = canonical;
    } else if (canonical != shared_label) {
      compact_labels = false;
    }
  };
  for (const Detection& detection : prediction.detections) {
    inspect_label(detection.label);
  }
  for (const Point& point : prediction.points) inspect_label(point.label);

  size_t index = 0;
  for (const Detection& detection : prediction.detections) {
    const cv::Scalar color = kAnnotationColors[index % kAnnotationColors.size()];
    const int left = std::clamp(static_cast<int>(std::lround(detection.box[0])),
                                0, image.cols - 1);
    const int top = std::clamp(static_cast<int>(std::lround(detection.box[1])),
                               0, image.rows - 1);
    const int right = std::clamp(static_cast<int>(std::lround(detection.box[2])),
                                 left + 1, image.cols);
    const int bottom = std::clamp(static_cast<int>(std::lround(detection.box[3])),
                                  top + 1, image.rows);
    const cv::Rect rectangle(left, top, right - left, bottom - top);
    cv::rectangle(image, rectangle, cv::Scalar(12, 15, 20), line_width + 2,
                  cv::LINE_AA);
    cv::rectangle(image, rectangle, color, line_width, cv::LINE_AA);
    const std::string caption =
        std::to_string(index + 1) +
        (compact_labels || detection.label.empty() ? "" : "  " + detection.label);
    DrawCaption(&image, caption, rectangle.tl(), color, font_scale, text_width,
                true);
    ++index;
  }
  for (const Point& point : prediction.points) {
    const cv::Scalar color = kAnnotationColors[index % kAnnotationColors.size()];
    const cv::Point center(
        std::clamp(static_cast<int>(std::lround(point.point[0])), 0, image.cols - 1),
        std::clamp(static_cast<int>(std::lround(point.point[1])), 0, image.rows - 1));
    const int marker_size = std::clamp(short_edge / 24, 18, 42);
    cv::drawMarker(image, center, cv::Scalar(12, 15, 20), cv::MARKER_CROSS,
                   marker_size + 4, line_width + 3, cv::LINE_AA);
    cv::drawMarker(image, center, color, cv::MARKER_CROSS, marker_size,
                   line_width, cv::LINE_AA);
    cv::circle(image, center, line_width + 4, cv::Scalar(12, 15, 20),
               cv::FILLED, cv::LINE_AA);
    cv::circle(image, center, line_width + 1, color, cv::FILLED, cv::LINE_AA);
    cv::circle(image, center, std::max(1, line_width / 2), cv::Scalar(255, 255, 255),
               cv::FILLED, cv::LINE_AA);
    const std::string caption =
        std::to_string(index + 1) +
        (compact_labels || point.label.empty() ? "" : "  " + point.label);
    const int offset = marker_size / 2 + line_width + 4;
    const int caption_x = center.x + offset < image.cols
                              ? center.x + offset
                              : std::max(0, center.x - offset);
    DrawCaption(&image, caption, cv::Point(caption_x, center.y + offset), color,
                font_scale, text_width, false);
    ++index;
  }
  if (compact_labels) {
    const std::string kind = prediction.points.empty() ? "targets" : "points";
    const std::string summary = std::to_string(target_count) + "  " + kind +
                                "  |  " + shared_label;
    const int margin = std::max(8, line_width * 4);
    DrawCaption(&image, summary,
                cv::Point(image.cols - margin, image.rows - margin),
                kAnnotationColors.front(), font_scale, text_width, true);
  }
  return image;
}

std::string Postprocessor::ToJson(const Prediction& prediction,
                                  const std::string& task,
                                  const std::string& stop_reason,
                                  uint64_t frame_index,
                                  const InferenceMetrics& metrics,
                                  bool pretty) const {
  std::ostringstream output;
  output << std::fixed << std::setprecision(3);
  const char* colon = pretty ? ": " : ":";
  const char* comma = pretty ? ", " : ",";
  const auto newline = [&]() {
    if (pretty) output << '\n';
  };
  const auto indent = [&](int level) {
    if (pretty) output << std::string(static_cast<size_t>(level * 2), ' ');
  };

  output << '{';
  newline();
  indent(1);
  output << "\"frame_index\"" << colon << frame_index << ',';
  newline();
  indent(1);
  output << "\"task\"" << colon << '"' << JsonEscape(task) << "\",";
  newline();
  indent(1);
  output << "\"stop_reason\"" << colon << '"' << JsonEscape(stop_reason)
         << "\",";
  newline();
  indent(1);
  output << "\"detections\"" << colon << '[';
  if (!prediction.detections.empty()) newline();
  for (size_t index = 0; index < prediction.detections.size(); ++index) {
    const Detection& item = prediction.detections[index];
    indent(2);
    output << "{\"label\"" << colon << '"' << JsonEscape(item.label)
           << '"' << comma << "\"bbox_xyxy\"" << colon << '[' << item.box[0]
           << comma << item.box[1] << comma << item.box[2] << comma << item.box[3]
           << "]}";
    if (index + 1 != prediction.detections.size()) output << ',';
    newline();
  }
  if (!prediction.detections.empty()) indent(1);
  output << "],";
  newline();
  indent(1);
  output << "\"points\"" << colon << '[';
  if (!prediction.points.empty()) newline();
  for (size_t index = 0; index < prediction.points.size(); ++index) {
    const Point& item = prediction.points[index];
    indent(2);
    output << "{\"label\"" << colon << '"' << JsonEscape(item.label)
           << '"' << comma << "\"point_xy\"" << colon << '[' << item.point[0]
           << comma << item.point[1] << "]}";
    if (index + 1 != prediction.points.size()) output << ',';
    newline();
  }
  if (!prediction.points.empty()) indent(1);
  output << "],";
  newline();

  const LanguageMetrics& language = metrics.language;
  const double tokens_per_second =
      language.decode_ms > 0.0
          ? static_cast<double>(language.generated_tokens) * 1000.0 /
                language.decode_ms
          : 0.0;
  indent(1);
  output << "\"generation\"" << colon << '{';
  newline();
  indent(2);
  output << "\"prompt_tokens\"" << colon << language.prompt_tokens << ',';
  newline();
  indent(2);
  output << "\"generated_tokens\"" << colon << language.generated_tokens << ',';
  newline();
  indent(2);
  output << "\"tokens_per_second\"" << colon << tokens_per_second << ',';
  newline();
  indent(2);
  output << "\"executed_mode\"" << colon << '"'
         << JsonEscape(language.executed_mode) << "\",";
  newline();
  indent(2);
  output << "\"fallback_reason\"" << colon << '"'
         << JsonEscape(language.fallback_reason) << "\",";
  newline();
  indent(2);
  output << "\"pbd_calls\"" << colon << language.pbd_calls << ',';
  newline();
  indent(2);
  output << "\"pbd_accepted_tokens\"" << colon << language.pbd_accepted_tokens
         << ',';
  newline();
  indent(2);
  output << "\"ar_calls\"" << colon << language.ar_calls << ',';
  newline();
  indent(2);
  output << "\"ar_tokens\"" << colon << language.ar_tokens << ',';
  newline();
  indent(2);
  output << "\"graph_timings\"" << colon << '[';
  if (!language.graph_timings.empty()) newline();
  for (size_t index = 0; index < language.graph_timings.size(); ++index) {
    const GraphTiming& timing = language.graph_timings[index];
    indent(3);
    output << "{\"graph\"" << colon << '"' << JsonEscape(timing.graph) << '"'
           << comma << "\"calls\"" << colon << timing.calls << comma
           << "\"total_ms\"" << colon << timing.total_ms << comma
           << "\"bpu_wait_ms\"" << colon << timing.bpu_wait_ms << comma
           << "\"submit_ms\"" << colon << timing.submit_ms << comma
           << "\"input_bytes\"" << colon << timing.input_bytes << comma
           << "\"output_bytes\"" << colon << timing.output_bytes << '}';
    if (index + 1 != language.graph_timings.size()) output << ',';
    newline();
  }
  if (!language.graph_timings.empty()) indent(2);
  output << "],";
  newline();
  indent(2);
  output << "\"cache_update_ms\"" << colon << language.cache_update_ms << ',';
  newline();
  indent(2);
  output << "\"host_decode_ms\"" << colon << language.host_decode_ms;
  newline();
  indent(1);
  output << "},";
  newline();

  indent(1);
  output << "\"timing_ms\"" << colon << '{';
  newline();
  indent(2);
  output << "\"preprocess\"" << colon << metrics.preprocess_ms << ',';
  newline();
  indent(2);
  output << "\"vision\"" << colon << metrics.vision_ms << ',';
  newline();
  indent(2);
  output << "\"language_prefill\"" << colon << language.prefill_ms << ',';
  newline();
  indent(2);
  output << "\"language_decode\"" << colon << language.decode_ms << ',';
  newline();
  indent(2);
  output << "\"language\"" << colon << metrics.language_ms << ',';
  newline();
  indent(2);
  output << "\"postprocess\"" << colon << metrics.postprocess_ms << ',';
  newline();
  indent(2);
  output << "\"total\"" << colon << metrics.total_ms;
  newline();
  indent(1);
  output << '}';
  newline();
  output << '}';
  return output.str();
}

}  // namespace locateanything

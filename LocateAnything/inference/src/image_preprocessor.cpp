#include "image_preprocessor.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include <opencv2/imgproc.hpp>

namespace locateanything {
namespace {

constexpr int kImageSize = 672;
constexpr int kPatchSize = 14;
constexpr int kGridSize = kImageSize / kPatchSize;
constexpr int kPrecisionBits = 22;

struct Coefficients {
  std::vector<int> starts;
  std::vector<std::vector<int32_t>> weights;
};

double Bicubic(double value) {
  value = std::abs(value);
  if (value < 1.0) return ((1.5 * value - 2.5) * value) * value + 1.0;
  if (value < 2.0) return (((-0.5 * value + 2.5) * value - 4.0) * value) + 2.0;
  return 0.0;
}

Coefficients BuildCoefficients(int input_size, int output_size) {
  const double scale = static_cast<double>(input_size) / output_size;
  const double filter_scale = std::max(1.0, scale);
  const double support = 2.0 * filter_scale;
  Coefficients coefficients;
  coefficients.starts.resize(output_size);
  coefficients.weights.resize(output_size);
  for (int output = 0; output < output_size; ++output) {
    const double center = (output + 0.5) * scale;
    int start = static_cast<int>(center - support + 0.5);
    int end = static_cast<int>(center + support + 0.5);
    start = std::max(0, start);
    end = std::min(input_size, end);
    std::vector<double> floating(static_cast<size_t>(std::max(0, end - start)));
    double sum = 0.0;
    for (int input = start; input < end; ++input) {
      const double weight = Bicubic((input - center + 0.5) / filter_scale);
      floating[static_cast<size_t>(input - start)] = weight;
      sum += weight;
    }
    coefficients.starts[output] = start;
    coefficients.weights[output].reserve(floating.size());
    for (double weight : floating) {
      const double normalized = sum == 0.0 ? 0.0 : weight / sum;
      coefficients.weights[output].push_back(
          static_cast<int32_t>(normalized * (1 << kPrecisionBits) + 0.5));
    }
  }
  return coefficients;
}

uint8_t ClipByte(int64_t value) {
  value >>= kPrecisionBits;
  return static_cast<uint8_t>(std::clamp<int64_t>(value, 0, 255));
}

cv::Mat PillowBicubicResize(const cv::Mat& source, int width, int height) {
  const Coefficients horizontal = BuildCoefficients(source.cols, width);
  cv::Mat intermediate(source.rows, width, CV_8UC3);
  for (int y = 0; y < source.rows; ++y) {
    const auto* input = source.ptr<cv::Vec3b>(y);
    auto* output = intermediate.ptr<cv::Vec3b>(y);
    for (int x = 0; x < width; ++x) {
      for (int channel = 0; channel < 3; ++channel) {
        int64_t value = int64_t{1} << (kPrecisionBits - 1);
        const auto& weights = horizontal.weights[x];
        for (size_t index = 0; index < weights.size(); ++index) {
          value += static_cast<int64_t>(
                       input[horizontal.starts[x] + static_cast<int>(index)][channel]) *
                   weights[index];
        }
        output[x][channel] = ClipByte(value);
      }
    }
  }

  const Coefficients vertical = BuildCoefficients(source.rows, height);
  cv::Mat resized(height, width, CV_8UC3);
  for (int y = 0; y < height; ++y) {
    auto* output = resized.ptr<cv::Vec3b>(y);
    const auto& weights = vertical.weights[y];
    for (int x = 0; x < width; ++x) {
      for (int channel = 0; channel < 3; ++channel) {
        int64_t value = int64_t{1} << (kPrecisionBits - 1);
        for (size_t index = 0; index < weights.size(); ++index) {
          value += static_cast<int64_t>(intermediate.ptr<cv::Vec3b>(
                       vertical.starts[y] + static_cast<int>(index))[x][channel]) *
                   weights[index];
        }
        output[x][channel] = ClipByte(value);
      }
    }
  }
  return resized;
}

uint16_t FloatToHalf(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16U) & 0x8000U;
  int32_t exponent = static_cast<int32_t>((bits >> 23U) & 0xffU) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffU;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000U) >> static_cast<uint32_t>(1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x1000U) >> 13U));
  }
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10U) |
                               ((mantissa + 0x1000U) >> 13U));
}

}  // namespace

PreparedImage ImagePreprocessor::Prepare(const cv::Mat& bgr) const {
  if (bgr.empty() || bgr.channels() != 3) {
    throw std::invalid_argument("input image must be a non-empty three-channel image");
  }

  const float scale = std::min(static_cast<float>(kImageSize) / bgr.cols,
                               static_cast<float>(kImageSize) / bgr.rows);
  const int resized_width = std::clamp(
      static_cast<int>(std::lround(bgr.cols * scale)), 1, kImageSize);
  const int resized_height = std::clamp(
      static_cast<int>(std::lround(bgr.rows * scale)), 1, kImageSize);
  const int left = (kImageSize - resized_width) / 2;
  const int top = (kImageSize - resized_height) / 2;

  const cv::Mat resized = PillowBicubicResize(bgr, resized_width, resized_height);
  cv::Mat canvas(kImageSize, kImageSize, CV_8UC3, cv::Scalar(128, 128, 128));
  resized.copyTo(canvas(cv::Rect(left, top, resized_width, resized_height)));

  PreparedImage output;
  output.transform = {bgr.cols,
                      bgr.rows,
                      resized_width,
                      resized_height,
                      left,
                      top,
                      static_cast<float>(resized_width) / bgr.cols,
                      static_cast<float>(resized_height) / bgr.rows};
  output.patches.reserve(static_cast<size_t>(kGridSize * kGridSize * 3 *
                                              kPatchSize * kPatchSize));
  for (int grid_y = 0; grid_y < kGridSize; ++grid_y) {
    for (int grid_x = 0; grid_x < kGridSize; ++grid_x) {
      for (int channel = 2; channel >= 0; --channel) {
        for (int patch_y = 0; patch_y < kPatchSize; ++patch_y) {
          const auto* row = canvas.ptr<cv::Vec3b>(grid_y * kPatchSize + patch_y);
          for (int patch_x = 0; patch_x < kPatchSize; ++patch_x) {
            const float value =
                static_cast<float>(row[grid_x * kPatchSize + patch_x][channel]) /
                    127.5f -
                1.0f;
            output.patches.push_back(FloatToHalf(value));
          }
        }
      }
    }
  }
  return output;
}

}  // namespace locateanything

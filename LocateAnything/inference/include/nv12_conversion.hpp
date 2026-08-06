#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace locateanything {

inline std::vector<uint8_t> BgrToNv12(const cv::Mat& input) {
  if (input.empty()) throw std::runtime_error("input image is empty");

  const int width = input.cols & ~1;
  const int height = input.rows & ~1;
  if (width <= 0 || height <= 0) {
    throw std::runtime_error("input image must be at least 2x2 pixels");
  }

  const cv::Mat bgr = input(cv::Rect(0, 0, width, height));
  cv::Mat i420;
  cv::cvtColor(bgr, i420, cv::COLOR_BGR2YUV_I420);

  const size_t y_size = static_cast<size_t>(width) * height;
  const size_t uv_plane_size = y_size / 4;
  std::vector<uint8_t> nv12(y_size + uv_plane_size * 2);
  std::copy_n(i420.data, y_size, nv12.data());

  const uint8_t* u = i420.data + y_size;
  const uint8_t* v = u + uv_plane_size;
  for (size_t i = 0; i < uv_plane_size; ++i) {
    nv12[y_size + i * 2] = u[i];
    nv12[y_size + i * 2 + 1] = v[i];
  }
  return nv12;
}

}  // namespace locateanything

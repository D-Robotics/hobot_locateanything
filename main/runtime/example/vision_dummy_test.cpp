// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Phase 1 sanity test for the C++ host runtime.
//
// Loads vision.hbm, feeds a dummy (1, 1024, 588) fp32 patch tensor, runs the
// "visual" graph once, prints input/output shape/dtype and a small summary
// of the output values (min/max/mean/NaN-count).
//
// Build (on S600):
//   mkdir build && cd build
//   cmake ..
//   make -j
//   ./vision_dummy_test <path-to-vision.hbm>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

#include "locateanything_runtime/hbm_session.hpp"
#include "locateanything_runtime/image_preprocess.hpp"

namespace rt = locateanything_runtime;

namespace {

void PrintTensorSummary(const char *label, const rt::Tensor &t) {
  std::printf("  %-20s shape=[", label);
  for (size_t i = 0; i < t.shape.size(); ++i) {
    std::printf("%d%s", t.shape[i], i + 1 == t.shape.size() ? "" : ",");
  }
  std::printf("] dtype=%s bytes=%zu\n",
              rt::DtypeName(t.dtype), t.data.size());

  // We know vision output is fp16. Decode and summarize.
  if (t.dtype == 4 /*F16*/ && !t.data.empty()) {
    // fp16 decoding on aarch64 without <stdfloat>: cast fp16 bits -> float
    // via a tiny software decoder. (We can't reinterpret as _Float16 here.)
    size_t n = t.data.size() / 2;
    const uint16_t *bits = reinterpret_cast<const uint16_t *>(t.data.data());
    float mn = 1e30f, mx = -1e30f, sum = 0;
    int nan_count = 0;
    for (size_t i = 0; i < n; ++i) {
      uint16_t h = bits[i];
      // sign(1) exponent(5) mantissa(10)
      uint32_t sign = (h >> 15) & 0x1;
      uint32_t exp = (h >> 10) & 0x1f;
      uint32_t mant = h & 0x3ff;
      float f;
      if (exp == 0) {
        if (mant == 0) {
          f = sign ? -0.0f : 0.0f;
        } else {
          // subnormal
          float val = mant / 1024.0f * (1.0f / 16384.0f);
          f = sign ? -val : val;
        }
      } else if (exp == 31) {
        // inf or nan
        nan_count++;
        f = std::nanf("");
      } else {
        float val = std::ldexp(1.0f + mant / 1024.0f,
                               static_cast<int>(exp) - 15);
        f = sign ? -val : val;
      }
      if (!std::isnan(f)) {
        if (f < mn) mn = f;
        if (f > mx) mx = f;
      }
      sum += std::isnan(f) ? 0.0f : f;
    }
    std::printf("    -> %zu fp16 values, min=%.4f max=%.4f mean=%.4f nan=%d\n",
                n, mn, mx, sum / static_cast<float>(n), nan_count);
  }
}

}  // namespace

int main(int argc, char **argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <vision.hbm>\n", argv[0]);
    return 1;
  }
  const std::string hbm_path = argv[1];

  std::printf("[vision_dummy_test] Phase-1 sanity\n");
  std::printf("[vision_dummy_test] hbm: %s\n", hbm_path.c_str());

  // 1) Load hbm.
  rt::HbmSession session;
  rt::Result r = session.Load(hbm_path);
  if (!r.ok()) {
    std::fprintf(stderr, "[FAIL] Load: code=%d msg=%s\n", r.code, r.message.c_str());
    return 2;
  }
  std::printf("[ok] Load. graphs in hbm: [");
  for (size_t i = 0; i < session.GetGraphNames().size(); ++i) {
    std::printf("%s%s", session.GetGraphNames()[i].c_str(),
                i + 1 == session.GetGraphNames().size() ? "" : ", ");
  }
  std::printf("]\n");

  // 2) Pick the "visual" graph.
  rt::Graph *g = session.GetGraph("visual");
  if (g == nullptr) {
    std::fprintf(stderr, "[FAIL] graph 'visual' not found in %s\n", hbm_path.c_str());
    return 3;
  }
  std::printf("[ok] graph visual: %d inputs, %d outputs\n",
              static_cast<int>(g->GetInputNames().size()),
              static_cast<int>(g->GetOutputNames().size()));

  // 3) Build dummy input.
  rt::ImagePatchTensor patch;
  rt::BuildDummyVisionPatchTensor(&patch);
  std::printf("[ok] dummy input built: shape=[1, %d, %d] floats=%zu bytes=%zu\n",
              1024, 588,
              patch.data.size(), patch.data.size() * sizeof(float));

  // 4) Pack into a Tensor that hbm_session expects (fp32, raw bytes).
  rt::Tensor in;
  in.shape = patch.shape;
  in.dtype = 7;  // HB_DNN_TENSOR_TYPE_F32
  in.data.resize(patch.data.size() * sizeof(float));
  std::memcpy(in.data.data(), patch.data.data(), in.data.size());

  // 5) Execute via the session's by-name convenience method (Graph keeps
  // its own C handle internally after GetGraph cached it).
  std::vector<rt::Tensor> outputs;
  r = session.ExecuteGraphByName("visual", {in}, &outputs);
  if (!r.ok()) {
    std::fprintf(stderr, "[FAIL] Execute: code=%d msg=%s\n", r.code, r.message.c_str());
    return 4;
  }

  // 6) Summarize outputs.
  std::printf("[ok] Execute returned %zu output tensors:\n", outputs.size());
  for (size_t i = 0; i < outputs.size(); ++i) {
    std::string label = "out[" + std::to_string(i) + "]";
    PrintTensorSummary(label.c_str(), outputs[i]);
  }

  std::printf("[verdict] vision.hbm Phase-1 sanity PASSED\n");
  return 0;
}

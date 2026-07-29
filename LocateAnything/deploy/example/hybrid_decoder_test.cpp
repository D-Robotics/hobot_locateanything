#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "locateanything_runtime/hybrid_decoder.hpp"

namespace rt = locateanything_runtime;

namespace {

constexpr int32_t kVocab = 152681;
constexpr int32_t kBoxStart = 151668;
constexpr int32_t kBoxEnd = 151669;
constexpr int32_t kCoordStart = 151677;

uint16_t FloatToFp16(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exponent = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mantissa = bits & 0x7fffffu;
  if (exponent <= 0) {
    if (exponent < -10) return static_cast<uint16_t>(sign);
    mantissa = (mantissa | 0x800000u) >> (1 - exponent);
    return static_cast<uint16_t>(sign | ((mantissa + 0x1000u) >> 13));
  }
  if (exponent >= 31) return static_cast<uint16_t>(sign | 0x7c00u);
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exponent) << 10) |
                               ((mantissa + 0x1000u) >> 13));
}

void Set(rt::Tensor* tensor, int32_t row, int32_t token, float value) {
  auto* raw = reinterpret_cast<uint16_t*>(tensor->data.data());
  raw[static_cast<size_t>(row) * kVocab + static_cast<size_t>(token)] =
      FloatToFp16(value);
}

rt::Tensor ThresholdCase() {
  rt::Tensor logits;
  logits.shape = {1, 6, kVocab};
  logits.dtype = 4;
  logits.data.resize(static_cast<size_t>(6) * kVocab * sizeof(uint16_t));
  auto* raw = reinterpret_cast<uint16_t*>(logits.data.data());
  std::fill(raw, raw + static_cast<size_t>(6) * kVocab, FloatToFp16(-20.0f));

  Set(&logits, 0, kBoxStart, 5.0f);
  for (int32_t row = 1; row <= 4; ++row) {
    Set(&logits, row, kCoordStart + row * 10, 5.0f);
  }
  Set(&logits, 5, kBoxEnd, 0.0f);
  Set(&logits, 5, 100, 0.05f);
  for (int32_t token = 200; token < 300; ++token) {
    Set(&logits, 5, token, -3.0f);
  }
  return logits;
}

}  // namespace

int main() {
  const rt::Tensor logits = ThresholdCase();
  const std::vector<int32_t> generated{42};

  rt::PbdDecodeConfig legacy;
  legacy.temperature = 1.0f;
  legacy.top_p = 1.0f;
  legacy.repetition_penalty = 1.1f;
  const rt::HybridDecision legacy_decision = rt::DecodePbd(
      logits, generated, legacy);

  rt::PbdDiagnostics diagnostics;
  const rt::HybridDecision official_decision = rt::DecodePbd(
      logits, generated, rt::PbdDecodeConfig{}, &diagnostics);

  const bool ok = legacy_decision.type == "error_box" &&
                  official_decision.type == "coord_box" && diagnostics.valid &&
                  diagnostics.legacy_end_score < 0.2f &&
                  diagnostics.official_end_score >= 0.2f &&
                  diagnostics.retained_tokens[5] < kVocab;
  std::printf(
      "legacy=%s official=%s end_score=%.6f->%.6f retained_row5=%d\n",
      legacy_decision.type.c_str(), official_decision.type.c_str(),
      diagnostics.legacy_end_score, diagnostics.official_end_score,
      diagnostics.retained_tokens[5]);
  std::printf("%s\n", ok ? "[PASS] hybrid_decoder_test"
                           : "[FAIL] hybrid_decoder_test");
  return ok ? 0 : 1;
}

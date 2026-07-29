// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "locateanything_runtime/kv_cache_ring.hpp"

namespace rt = locateanything_runtime;

namespace {

bool RunCase(size_t cache_rows, size_t row_bytes,
             const std::vector<size_t>& commits) {
  std::vector<uint8_t> reference(cache_rows * row_bytes);
  for (size_t index = 0; index < reference.size(); ++index) {
    reference[index] = static_cast<uint8_t>((index * 17 + 3) & 0xff);
  }
  std::vector<uint8_t> ring = reference;
  size_t byte_offset = 0;
  uint8_t next_value = 101;

  for (size_t step = 0; step < commits.size(); ++step) {
    const size_t rows = commits[step];
    std::vector<uint8_t> update(rows * row_bytes);
    for (uint8_t& value : update) value = next_value++;

    std::memmove(reference.data(), reference.data() + rows * row_bytes,
                 (cache_rows - rows) * row_bytes);
    std::memcpy(reference.data() + (cache_rows - rows) * row_bytes,
                update.data(), update.size());

    uint64_t copied_bytes = 0;
    if (!rt::AppendMirroredRingRows(
            &ring, cache_rows, row_bytes, update.data(), update.size(), rows,
            &byte_offset, &copied_bytes)) {
      std::printf("[FAIL] append rejected at step=%zu\n", step);
      return false;
    }
    if (copied_bytes != update.size() * 2 ||
        byte_offset + reference.size() > ring.size() ||
        !std::equal(reference.begin(), reference.end(),
                    ring.begin() + static_cast<std::ptrdiff_t>(byte_offset))) {
      std::printf("[FAIL] mismatch at step=%zu rows=%zu offset=%zu\n",
                  step, rows, byte_offset);
      return false;
    }
  }
  return true;
}

}  // namespace

int main() {
  bool ok = true;
  ok = RunCase(8, 3, {1, 1, 2, 1, 3, 1, 4, 2}) && ok;
  ok = RunCase(7, 5, {6, 1, 3, 2, 7, 1}) && ok;
  std::printf("[verdict] kv_cache_ring test %s\n", ok ? "PASSED" : "FAILED");
  return ok ? 0 : 1;
}

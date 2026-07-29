// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "locateanything_runtime/hbm_session.hpp"

namespace locateanything_runtime {

// Advance a right-aligned KV cache without moving its historical rows.
// `storage` keeps two identical copies of the physical ring so the logical
// cache view is always one contiguous cache-sized span at `byte_offset`.
bool AppendMirroredRingRows(std::vector<uint8_t>* storage,
                            size_t cache_rows,
                            size_t row_bytes,
                            const uint8_t* update,
                            size_t update_bytes,
                            size_t committed_rows,
                            size_t* byte_offset,
                            uint64_t* copied_bytes = nullptr);

// Device-resident variant. Only the appended rows are written and cache-cleaned;
// the cache-sized logical view remains contiguous through mirrored storage.
bool AppendMirroredDeviceRingRows(Tensor* cache,
                                  size_t cache_rows,
                                  size_t row_bytes,
                                  const uint8_t* update,
                                  size_t update_bytes,
                                  size_t committed_rows,
                                  uint64_t* copied_bytes = nullptr);

}  // namespace locateanything_runtime

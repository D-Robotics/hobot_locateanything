// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include "runtime/hbm.hpp"

namespace locateanything_runtime {

/**
 * @brief Append host rows to a mirrored ring while preserving a contiguous view.
 *
 * The storage keeps two identical copies of the physical ring so callers can
 * expose a cache-sized logical span without moving historical rows.
 * @param storage Host storage containing one or two mirrored cache copies.
 * @param cache_rows Capacity of one cache copy in rows.
 * @param row_bytes Byte width of one cache row.
 * @param update Source rows to append.
 * @param update_bytes Available bytes at update.
 * @param committed_rows Number of leading update rows to commit.
 * @param byte_offset In/out logical start offset of the contiguous view.
 * @param copied_bytes Optional accumulated copy counter.
 * @return True when the update contract is valid and all rows were appended.
 */
bool AppendMirroredRingRows(std::vector<uint8_t>* storage,
                            size_t cache_rows,
                            size_t row_bytes,
                            const uint8_t* update,
                            size_t update_bytes,
                            size_t committed_rows,
                            size_t* byte_offset,
                            uint64_t* copied_bytes = nullptr);

/**
 * @brief Append rows directly to a device-backed mirrored ring cache.
 * @param cache Device-backed tensor and logical byte offset.
 * @param cache_rows Capacity of one cache copy in rows.
 * @param row_bytes Byte width of one cache row.
 * @param update Source rows to append.
 * @param update_bytes Available bytes at update.
 * @param committed_rows Number of leading update rows to commit.
 * @param copied_bytes Optional accumulated copy counter.
 * @return True when both mirrored ranges were written and cache-cleaned.
 */
bool AppendMirroredDeviceRingRows(Tensor* cache,
                                  size_t cache_rows,
                                  size_t row_bytes,
                                  const uint8_t* update,
                                  size_t update_bytes,
                                  size_t committed_rows,
                                  uint64_t* copied_bytes = nullptr);

}  // namespace locateanything_runtime

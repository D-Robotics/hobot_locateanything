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
 * @param[in,out] storage Host storage containing one or two mirrored cache
 * copies.
 * @param[in] cache_rows Capacity of one cache copy in rows.
 * @param[in] row_bytes Byte width of one cache row.
 * @param[in] update Source rows to append.
 * @param[in] update_bytes Available bytes at update.
 * @param[in] committed_rows Number of leading update rows to commit.
 * @param[in,out] byte_offset Logical start offset of the contiguous view.
 * @param[in,out] copied_bytes Optional accumulated copy counter.
 * @return True when the update contract is valid and all rows were appended.
 * Returns false for null pointers, zero or overflowing dimensions,
 * insufficient source/storage capacity, or a misaligned logical offset.
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
 * @param[in,out] cache Device-backed tensor and logical byte offset.
 * @param[in] cache_rows Capacity of one cache copy in rows.
 * @param[in] row_bytes Byte width of one cache row.
 * @param[in] update Source rows to append.
 * @param[in] update_bytes Available bytes at update.
 * @param[in] committed_rows Number of leading update rows to commit.
 * @param[in,out] copied_bytes Optional accumulated copy counter.
 * @return True when both mirrored ranges were written and cache-cleaned.
 * Returns false for an invalid tensor/storage contract or a device write
 * failure. A device failure can leave rows written before the failure in
 * place, while the logical byte offset remains unchanged.
 */
bool AppendMirroredDeviceRingRows(Tensor* cache,
                                  size_t cache_rows,
                                  size_t row_bytes,
                                  const uint8_t* update,
                                  size_t update_bytes,
                                  size_t committed_rows,
                                  uint64_t* copied_bytes = nullptr);

}  // namespace locateanything_runtime

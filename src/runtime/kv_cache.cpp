// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include "runtime/kv_cache.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace locateanything_runtime {

bool AppendMirroredRingRows(std::vector<uint8_t>* storage,
                            size_t cache_rows,
                            size_t row_bytes,
                            const uint8_t* update,
                            size_t update_bytes,
                            size_t committed_rows,
                            size_t* byte_offset,
                            uint64_t* copied_bytes) {
  if (storage == nullptr || byte_offset == nullptr || update == nullptr ||
      cache_rows == 0 || row_bytes == 0 || committed_rows == 0 ||
      committed_rows > cache_rows ||
      cache_rows > std::numeric_limits<size_t>::max() / row_bytes) {
    return false;
  }
  const size_t cache_bytes = cache_rows * row_bytes;
  if (cache_bytes > std::numeric_limits<size_t>::max() / 2 ||
      committed_rows > std::numeric_limits<size_t>::max() / row_bytes) {
    return false;
  }
  const size_t update_required = committed_rows * row_bytes;
  if (update_bytes < update_required || *byte_offset % row_bytes != 0 ||
      *byte_offset >= cache_bytes) {
    return false;
  }

  if (storage->size() == cache_bytes) {
    if (*byte_offset != 0) return false;
    std::vector<uint8_t> mirrored(cache_bytes * 2);
    std::memcpy(mirrored.data(), storage->data(), cache_bytes);
    std::memcpy(mirrored.data() + cache_bytes, storage->data(), cache_bytes);
    *storage = std::move(mirrored);
    *byte_offset = 0;
  }
  if (storage->size() != cache_bytes * 2) return false;

  const size_t old_start = *byte_offset / row_bytes;
  const size_t new_start = (old_start + committed_rows) % cache_rows;
  size_t destination =
      (new_start + cache_rows - committed_rows) % cache_rows;
  size_t source_row = 0;
  size_t remaining = committed_rows;
  while (remaining > 0) {
    const size_t rows = std::min(remaining, cache_rows - destination);
    const size_t chunk_bytes = rows * row_bytes;
    std::memcpy(storage->data() + destination * row_bytes,
                update + source_row * row_bytes, chunk_bytes);
    std::memcpy(storage->data() + cache_bytes + destination * row_bytes,
                update + source_row * row_bytes, chunk_bytes);
    remaining -= rows;
    source_row += rows;
    destination = 0;
  }
  *byte_offset = new_start * row_bytes;
  if (copied_bytes != nullptr) {
    *copied_bytes += static_cast<uint64_t>(update_required) * 2;
  }
  return true;
}

bool AppendMirroredDeviceRingRows(Tensor* cache,
                                  size_t cache_rows,
                                  size_t row_bytes,
                                  const uint8_t* update,
                                  size_t update_bytes,
                                  size_t committed_rows,
                                  uint64_t* copied_bytes) {
  if (cache == nullptr || cache->device_buffer == nullptr || update == nullptr ||
      cache_rows == 0 || row_bytes == 0 || committed_rows == 0 ||
      committed_rows > cache_rows ||
      cache_rows > std::numeric_limits<size_t>::max() / row_bytes ||
      committed_rows > std::numeric_limits<size_t>::max() / row_bytes) {
    return false;
  }
  const size_t cache_bytes = cache_rows * row_bytes;
  const size_t update_required = committed_rows * row_bytes;
  if (cache_bytes > std::numeric_limits<size_t>::max() / 2 ||
      cache->device_buffer->size() != cache_bytes * 2 ||
      update_bytes < update_required || cache->byte_offset % row_bytes != 0 ||
      cache->byte_offset >= cache_bytes) {
    return false;
  }

  const size_t old_start = cache->byte_offset / row_bytes;
  const size_t new_start = (old_start + committed_rows) % cache_rows;
  size_t destination =
      (new_start + cache_rows - committed_rows) % cache_rows;
  size_t source_row = 0;
  size_t remaining = committed_rows;
  while (remaining > 0) {
    const size_t rows = std::min(remaining, cache_rows - destination);
    const size_t chunk_bytes = rows * row_bytes;
    const size_t destination_bytes = destination * row_bytes;
    if (!WriteDeviceBuffer(cache->device_buffer, destination_bytes,
                           update + source_row * row_bytes, chunk_bytes).ok() ||
        !WriteDeviceBuffer(cache->device_buffer, cache_bytes + destination_bytes,
                           update + source_row * row_bytes, chunk_bytes).ok()) {
      return false;
    }
    remaining -= rows;
    source_row += rows;
    destination = 0;
  }
  cache->byte_offset = new_start * row_bytes;
  if (copied_bytes != nullptr) {
    *copied_bytes += static_cast<uint64_t>(update_required) * 2;
  }
  return true;
}

}  // namespace locateanything_runtime

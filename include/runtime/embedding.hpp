// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Embedding lookup for the LocateAnything host runtime.
//
// Memory-maps the configured fp16 embedding table and gathers rows by token
// ID. Vocabulary and hidden dimensions are supplied from the validated
// Language HBM contract rather than duplicated here.
//
// Vendored flow from upstream `modeling_qwen2.py::Qwen2Model.get_input_embeddings`
// — the embed lookup itself is a plain index-gather, nothing LA-specific
// beyond the vocab size. We mmap rather than load to keep peak RSS low
// (597 MB virtual, only touched pages paged in).

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace locateanything_runtime {

class EmbedLookup {
 public:
  /** Create an unopened embedding lookup. */
  EmbedLookup() = default;
  /** Unmap the embedding file and close its descriptor. */
  ~EmbedLookup();

  EmbedLookup(const EmbedLookup &) = delete;
  EmbedLookup &operator=(const EmbedLookup &) = delete;

  /**
   * @brief Memory-map an embedding table and validate its minimum size.
   * @param[in] path Path to the fp16 row-major embedding file.
   * @param[in] vocab_size Number of vocabulary rows.
   * @param[in] hidden_dim Number of fp16 elements per row.
   * @return True when the file was opened, was large enough for all requested
   * rows, and was mapped successfully; false after any file or mapping error.
   * @pre vocab_size and hidden_dim are positive.
   */
  bool Open(const std::string &path, int32_t vocab_size, int32_t hidden_dim);

  /**
   * @brief Gather token rows into a caller-owned contiguous fp16 buffer.
   * @param[in] token_ids Token IDs to gather; invalid IDs map to row zero.
   * @param[in] count Number of token IDs and output rows.
   * @param[out] out Destination with room for count times hidden_dim fp16
   * values.
   * @pre Open succeeded, token_ids and out are non-null, count is non-negative,
   * and out has the documented capacity. If Open has not succeeded, the
   * function returns without writing output.
   */
  void Gather(const int32_t *token_ids, int32_t count, void *out) const;

  /** @brief Return the configured vocabulary size. @return Vocabulary rows. */
  int32_t VocabSize() const { return vocab_size_; }
  /** @brief Return the configured row width. @return Hidden dimension. */
  int32_t HiddenDim() const { return hidden_dim_; }
  /** @brief Check whether a file is mapped. @return True when Open succeeded. */
  bool IsOpen() const { return base_ != nullptr; }

 private:
  void *base_ = nullptr;      // mmap'd file base
  int64_t file_bytes_ = 0;    // total file size
  int32_t vocab_size_ = 0;
  int32_t hidden_dim_ = 0;
  int fd_ = -1;               // underlying file descriptor (kept for munmap)
};

}  // namespace locateanything_runtime

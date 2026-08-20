// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// Embedding lookup for the LocateAnything host runtime.
//
// Memory-maps the `LocateAnything-3B_embed_tokens.bin` file (597 MB,
// 152681 x 2048 fp16) and gathers rows by token ID. The gathered rows
// are returned as a contiguous fp16 buffer suitable for feeding into
// language.hbm's prefill/decode input_0 `(1, q_len, 2048) fp16`.
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
   * @param path Path to the fp16 row-major embedding file.
   * @param vocab_size Number of vocabulary rows.
   * @param hidden_dim Number of fp16 elements per row.
   * @return True when the file was opened and mapped successfully.
   */
  bool Open(const std::string &path, int32_t vocab_size, int32_t hidden_dim);

  /**
   * @brief Gather token rows into a caller-owned contiguous fp16 buffer.
   * @param token_ids Token IDs to gather; invalid IDs map to row zero.
   * @param count Number of token IDs and output rows.
   * @param out Destination with room for count times hidden_dim fp16 values.
   */
  void Gather(const int32_t *token_ids, int32_t count, void *out) const;

  /** Return the vocabulary row count configured by Open. */
  int32_t VocabSize() const { return vocab_size_; }
  /** Return the embedding hidden dimension configured by Open. */
  int32_t HiddenDim() const { return hidden_dim_; }
  /** Return whether an embedding file is currently mapped. */
  bool IsOpen() const { return base_ != nullptr; }

 private:
  void *base_ = nullptr;      // mmap'd file base
  int64_t file_bytes_ = 0;    // total file size
  int32_t vocab_size_ = 0;    // 152681
  int32_t hidden_dim_ = 0;    // 2048
  int fd_ = -1;               // underlying file descriptor (kept for munmap)
};

}  // namespace locateanything_runtime

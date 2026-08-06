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
  EmbedLookup() = default;
  ~EmbedLookup();

  EmbedLookup(const EmbedLookup &) = delete;
  EmbedLookup &operator=(const EmbedLookup &) = delete;

  // Open `embed_tokens.bin` and validate its size against
  // (vocab_size * hidden_dim * sizeof(fp16)). Returns false on mismatch.
  bool Open(const std::string &path, int32_t vocab_size, int32_t hidden_dim);

  // Gather `count` rows by the token IDs in `token_ids` (length `count`),
  // writing `count * hidden_dim * 2` bytes of fp16 into `out` (caller-
  // allocated). Out-of-range token IDs map to the row at index 0 (the
  // <pad>/<bos> row) rather than crashing — matches the upstream
  // `get_input_embeddings` fallback behaviour for safety.
  void Gather(const int32_t *token_ids, int32_t count, void *out) const;

  int32_t VocabSize() const { return vocab_size_; }
  int32_t HiddenDim() const { return hidden_dim_; }
  bool IsOpen() const { return base_ != nullptr; }

 private:
  void *base_ = nullptr;      // mmap'd file base
  int64_t file_bytes_ = 0;    // total file size
  int32_t vocab_size_ = 0;    // 152681
  int32_t hidden_dim_ = 0;    // 2048
  int fd_ = -1;               // underlying file descriptor (kept for munmap)
};

}  // namespace locateanything_runtime

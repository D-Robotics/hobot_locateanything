// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include "locateanything_runtime/embed_lookup.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <iostream>

namespace locateanything_runtime {

namespace {

constexpr int64_t kFp16Bytes = 2;  // sizeof(__fp16) on aarch64

}  // namespace

EmbedLookup::~EmbedLookup() {
  if (base_ != nullptr) {
    munmap(base_, file_bytes_);
    base_ = nullptr;
  }
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
}

bool EmbedLookup::Open(const std::string &path, int32_t vocab_size, int32_t hidden_dim) {
  // Close any prior mapping.
  if (base_ != nullptr) {
    munmap(base_, file_bytes_);
    base_ = nullptr;
  }
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }

  fd_ = open(path.c_str(), O_RDONLY);
  if (fd_ < 0) {
    std::cerr << "[EmbedLookup] open failed: " << path << " errno=" << errno
              << " (" << strerror(errno) << ")" << std::endl;
    return false;
  }

  struct stat st;
  if (fstat(fd_, &st) != 0) {
    std::cerr << "[EmbedLookup] fstat failed: " << path << std::endl;
    close(fd_);
    fd_ = -1;
    return false;
  }
  file_bytes_ = st.st_size;

  int64_t expected = static_cast<int64_t>(vocab_size) * hidden_dim * kFp16Bytes;
  if (file_bytes_ < expected) {
    std::cerr << "[EmbedLookup] file too small: " << path << " got "
              << file_bytes_ << " bytes, need " << expected
              << " (vocab=" << vocab_size << " hidden=" << hidden_dim << ")"
              << std::endl;
    close(fd_);
    fd_ = -1;
    return false;
  }
  // Allow file_bytes_ > expected (hbdk4 pads vocab 152681 -> 152704 for
  // 64-byte alignment when compiling). We only read the first `expected`
  // bytes; the rest is padding.

  void *p = mmap(nullptr, file_bytes_, PROT_READ, MAP_PRIVATE, fd_, 0);
  if (p == MAP_FAILED) {
    std::cerr << "[EmbedLookup] mmap failed: " << path << " errno=" << errno
              << " (" << strerror(errno) << ")" << std::endl;
    close(fd_);
    fd_ = -1;
    return false;
  }
  base_ = p;
  vocab_size_ = vocab_size;
  hidden_dim_ = hidden_dim;
  return true;
}

void EmbedLookup::Gather(const int32_t *token_ids, int32_t count, void *out) const {
  if (base_ == nullptr) {
    return;
  }
  const uint8_t *src = static_cast<const uint8_t *>(base_);
  uint8_t *dst = static_cast<uint8_t *>(out);
  const int64_t row_bytes = static_cast<int64_t>(hidden_dim_) * kFp16Bytes;

  for (int32_t i = 0; i < count; ++i) {
    int32_t id = token_ids[i];
    // Out-of-range -> clamp to row 0 (<pad>/<bos>). Matches upstream
    // get_input_embeddings safe-fallback.
    if (id < 0 || id >= vocab_size_) {
      id = 0;
    }
    const uint8_t *row = src + static_cast<int64_t>(id) * row_bytes;
    std::memcpy(dst + static_cast<int64_t>(i) * row_bytes, row, row_bytes);
  }
}

}  // namespace locateanything_runtime

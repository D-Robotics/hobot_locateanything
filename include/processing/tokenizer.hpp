#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace locateanything {

class Tokenizer {
 public:
  /** Create an empty tokenizer; call Load before Encode or Decode. */
  Tokenizer();
  /** Release tokenizer vocabulary and merge tables. */
  ~Tokenizer();
  /** Move-construct a tokenizer without copying its tables. */
  Tokenizer(Tokenizer&&) noexcept;
  /** Move-assign a tokenizer without copying its tables. */
  Tokenizer& operator=(Tokenizer&&) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  /**
   * @brief Load tokenizer vocabulary, added tokens, and merge ranks.
   * @param directory Directory containing tokenizer asset files.
   */
  void Load(const std::string& directory);
  /**
   * @brief Encode UTF-8 text into model token IDs.
   * @param text Input prompt text.
   * @return Encoded model token IDs.
   */
  std::vector<int32_t> Encode(const std::string& text) const;
  /**
   * @brief Decode model token IDs into UTF-8 text.
   * @param tokens Model token IDs.
   * @return Decoded UTF-8 text.
   */
  std::string Decode(const std::vector<int32_t>& tokens) const;
  /**
   * @brief Resolve one special or vocabulary token to its numeric ID.
   * @param token Exact token text.
   * @return Token ID, or -1 when absent.
   */
  int32_t TokenId(const std::string& token) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything

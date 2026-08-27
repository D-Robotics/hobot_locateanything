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
  /**
   * @brief Move-construct a tokenizer without copying its tables.
   * @param other Tokenizer whose tables are transferred.
   */
  Tokenizer(Tokenizer&& other) noexcept;
  /**
   * @brief Move-assign a tokenizer without copying its tables.
   * @param other Tokenizer whose tables are transferred.
   * @return This tokenizer after replacing its previous tables.
   */
  Tokenizer& operator=(Tokenizer&& other) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  /**
   * @brief Load tokenizer vocabulary, added tokens, and merge ranks.
   * @param[in] directory Directory containing tokenizer asset files.
   * @throws std::runtime_error if an asset is missing or malformed.
   */
  void Load(const std::string& directory);
  /**
   * @brief Encode UTF-8 text into model token IDs.
   * @param[in] text Input prompt text.
   * @return Encoded model token IDs.
   * @throws std::logic_error if the tokenizer is not loaded.
   * @throws std::runtime_error if input cannot be represented by the vocabulary.
   */
  std::vector<int32_t> Encode(const std::string& text) const;
  /**
   * @brief Decode model token IDs into UTF-8 text.
   * @param[in] tokens Model token IDs; IDs absent from the loaded vocabulary
   * are ignored.
   * @return Decoded UTF-8 text for all recognized IDs.
   * @throws std::runtime_error if recognized byte tokens cannot be decoded.
   */
  std::string Decode(const std::vector<int32_t>& tokens) const;
  /**
   * @brief Resolve one special or vocabulary token to its numeric ID.
   * @param[in] token Exact token text.
   * @return Token ID, or -1 when absent.
   */
  int32_t TokenId(const std::string& token) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace locateanything {

class Tokenizer {
 public:
  Tokenizer();
  ~Tokenizer();
  Tokenizer(Tokenizer&&) noexcept;
  Tokenizer& operator=(Tokenizer&&) noexcept;
  Tokenizer(const Tokenizer&) = delete;
  Tokenizer& operator=(const Tokenizer&) = delete;

  void Load(const std::string& directory);
  std::vector<int32_t> Encode(const std::string& text) const;
  std::string Decode(const std::vector<int32_t>& tokens) const;
  int32_t TokenId(const std::string& token) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace locateanything

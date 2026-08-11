#include "processing/tokenizer.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace locateanything {
namespace {

/**
 * @brief Read a tokenizer asset as an unmodified binary string.
 * @param path Tokenizer asset path.
 * @return Complete file contents.
 */
std::string ReadText(const std::string& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open tokenizer asset: " + path);
  return std::string(std::istreambuf_iterator<char>(stream),
                     std::istreambuf_iterator<char>());
}

/**
 * @brief Append one Unicode scalar value encoded as UTF-8.
 * @param codepoint Unicode scalar value.
 * @param output Destination UTF-8 string.
 */
void AppendUtf8(uint32_t codepoint, std::string* output) {
  if (codepoint <= 0x7fU) {
    output->push_back(static_cast<char>(codepoint));
  } else if (codepoint <= 0x7ffU) {
    output->push_back(static_cast<char>(0xc0U | (codepoint >> 6U)));
    output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else if (codepoint <= 0xffffU) {
    output->push_back(static_cast<char>(0xe0U | (codepoint >> 12U)));
    output->push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  } else {
    output->push_back(static_cast<char>(0xf0U | (codepoint >> 18U)));
    output->push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3fU)));
    output->push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3fU)));
    output->push_back(static_cast<char>(0x80U | (codepoint & 0x3fU)));
  }
}

/**
 * @brief Parse four hexadecimal digits from a JSON Unicode escape.
 * @param text Complete JSON text.
 * @param offset Offset of the first hexadecimal digit.
 * @return Parsed 16-bit code unit.
 */
uint32_t ReadHex4(const std::string& text, size_t offset) {
  uint32_t value = 0;
  for (size_t index = 0; index < 4; ++index) {
    const char item = text.at(offset + index);
    value <<= 4U;
    if (item >= '0' && item <= '9') value |= static_cast<uint32_t>(item - '0');
    else if (item >= 'a' && item <= 'f') value |= static_cast<uint32_t>(item - 'a' + 10);
    else if (item >= 'A' && item <= 'F') value |= static_cast<uint32_t>(item - 'A' + 10);
    else throw std::runtime_error("invalid JSON unicode escape");
  }
  return value;
}

/**
 * @brief Parse one JSON string including escapes and surrogate pairs.
 * @param text Complete JSON text.
 * @param offset In/out parser cursor positioned at the opening quote.
 * @return Decoded UTF-8 string.
 */
std::string ParseJsonString(const std::string& text, size_t* offset) {
  if (*offset >= text.size() || text[*offset] != '"') {
    throw std::runtime_error("invalid tokenizer JSON string");
  }
  ++*offset;
  std::string value;
  while (*offset < text.size()) {
    const char item = text[(*offset)++];
    if (item == '"') return value;
    if (item != '\\') {
      value.push_back(item);
      continue;
    }
    if (*offset >= text.size()) throw std::runtime_error("truncated JSON escape");
    const char escaped = text[(*offset)++];
    switch (escaped) {
      case '"': value.push_back('"'); break;
      case '\\': value.push_back('\\'); break;
      case '/': value.push_back('/'); break;
      case 'b': value.push_back('\b'); break;
      case 'f': value.push_back('\f'); break;
      case 'n': value.push_back('\n'); break;
      case 'r': value.push_back('\r'); break;
      case 't': value.push_back('\t'); break;
      case 'u': {
        uint32_t codepoint = ReadHex4(text, *offset);
        *offset += 4;
        if (codepoint >= 0xd800U && codepoint <= 0xdbffU &&
            *offset + 6 <= text.size() && text[*offset] == '\\' &&
            text[*offset + 1] == 'u') {
          const uint32_t low = ReadHex4(text, *offset + 2);
          if (low >= 0xdc00U && low <= 0xdfffU) {
            codepoint = 0x10000U + ((codepoint - 0xd800U) << 10U) +
                        (low - 0xdc00U);
            *offset += 6;
          }
        }
        AppendUtf8(codepoint, &value);
        break;
      }
      default: throw std::runtime_error("unsupported JSON escape");
    }
  }
  throw std::runtime_error("unterminated tokenizer JSON string");
}

/**
 * @brief Parse a compact tokenizer string-to-integer JSON object.
 * @param path JSON asset path.
 * @return Parsed token-to-ID mapping.
 */
std::unordered_map<std::string, int32_t> ParseStringIntMap(
    const std::string& path) {
  const std::string text = ReadText(path);
  size_t offset = text.find('{');
  if (offset == std::string::npos) throw std::runtime_error("invalid tokenizer JSON");
  ++offset;
  std::unordered_map<std::string, int32_t> result;
  while (offset < text.size()) {
    while (offset < text.size() &&
           (std::isspace(static_cast<unsigned char>(text[offset])) ||
            text[offset] == ',')) {
      ++offset;
    }
    if (offset >= text.size() || text[offset] == '}') break;
    const std::string key = ParseJsonString(text, &offset);
    while (offset < text.size() && std::isspace(static_cast<unsigned char>(text[offset]))) {
      ++offset;
    }
    if (offset >= text.size() || text[offset++] != ':') {
      throw std::runtime_error("invalid tokenizer JSON object");
    }
    while (offset < text.size() && std::isspace(static_cast<unsigned char>(text[offset]))) {
      ++offset;
    }
    bool negative = false;
    if (offset < text.size() && text[offset] == '-') {
      negative = true;
      ++offset;
    }
    int64_t value = 0;
    const size_t number_start = offset;
    while (offset < text.size() && std::isdigit(static_cast<unsigned char>(text[offset]))) {
      value = value * 10 + (text[offset++] - '0');
    }
    if (number_start == offset || value > std::numeric_limits<int32_t>::max()) {
      throw std::runtime_error("invalid tokenizer ID");
    }
    result.emplace(key, static_cast<int32_t>(negative ? -value : value));
  }
  return result;
}

/**
 * @brief Decode UTF-8 into scalar values for tokenizer byte conversion.
 * @param value Valid UTF-8 input.
 * @return Unicode scalar values.
 */
std::vector<uint32_t> Utf8Codepoints(const std::string& value) {
  std::vector<uint32_t> output;
  for (size_t index = 0; index < value.size();) {
    const uint8_t first = static_cast<uint8_t>(value[index]);
    uint32_t codepoint = first;
    size_t length = 1;
    if ((first & 0xe0U) == 0xc0U) {
      codepoint = first & 0x1fU;
      length = 2;
    } else if ((first & 0xf0U) == 0xe0U) {
      codepoint = first & 0x0fU;
      length = 3;
    } else if ((first & 0xf8U) == 0xf0U) {
      codepoint = first & 0x07U;
      length = 4;
    }
    if (index + length > value.size()) throw std::runtime_error("invalid UTF-8 tokenizer token");
    for (size_t byte = 1; byte < length; ++byte) {
      const uint8_t continuation = static_cast<uint8_t>(value[index + byte]);
      if ((continuation & 0xc0U) != 0x80U) {
        throw std::runtime_error("invalid UTF-8 tokenizer token");
      }
      codepoint = (codepoint << 6U) | (continuation & 0x3fU);
    }
    output.push_back(codepoint);
    index += length;
  }
  return output;
}

/**
 * @brief Split UTF-8 into byte-aligned codepoint substrings.
 * @param value Valid UTF-8 input.
 * @return One substring per encoded codepoint.
 */
std::vector<std::string> SplitUtf8(const std::string& value) {
  std::vector<std::string> output;
  for (size_t index = 0; index < value.size();) {
    const uint8_t first = static_cast<uint8_t>(value[index]);
    size_t length = 1;
    if ((first & 0xe0U) == 0xc0U) length = 2;
    else if ((first & 0xf0U) == 0xe0U) length = 3;
    else if ((first & 0xf8U) == 0xf0U) length = 4;
    if (index + length > value.size()) throw std::runtime_error("invalid UTF-8 input");
    output.emplace_back(value.substr(index, length));
    index += length;
  }
  return output;
}

/**
 * @brief Classify letters for the model's byte-level pretokenization.
 * @param item Input byte.
 * @return True for ASCII letters or a non-ASCII leading/continuation byte.
 */
bool IsLetterByte(unsigned char item) {
  return std::isalpha(item) != 0 || item >= 0x80U;
}

/**
 * @brief Apply tokenizer whitespace, letter, digit, and punctuation rules.
 * @param text UTF-8 prompt segment without added tokens.
 * @return Pretokenized byte strings for BPE merging.
 */
std::vector<std::string> Pretokenize(const std::string& text) {
  std::vector<std::string> pieces;
  for (size_t index = 0; index < text.size();) {
    const unsigned char item = static_cast<unsigned char>(text[index]);
    const size_t start = index;
    if (item == '\r' || item == '\n') {
      while (index < text.size() && (text[index] == '\r' || text[index] == '\n')) ++index;
    } else if (std::isspace(item)) {
      while (index < text.size() && text[index] != '\r' && text[index] != '\n' &&
             std::isspace(static_cast<unsigned char>(text[index]))) {
        ++index;
      }
      if (index - start == 1 && index < text.size() &&
          IsLetterByte(static_cast<unsigned char>(text[index]))) {
        while (index < text.size() &&
               IsLetterByte(static_cast<unsigned char>(text[index]))) {
          ++index;
        }
      } else if (index - start == 1 && index < text.size() &&
                 !std::isspace(static_cast<unsigned char>(text[index])) &&
                 !IsLetterByte(static_cast<unsigned char>(text[index])) &&
                 !std::isdigit(static_cast<unsigned char>(text[index]))) {
        while (index < text.size() &&
               !std::isspace(static_cast<unsigned char>(text[index])) &&
               !IsLetterByte(static_cast<unsigned char>(text[index])) &&
               !std::isdigit(static_cast<unsigned char>(text[index]))) {
          ++index;
        }
      }
    } else if (IsLetterByte(item)) {
      while (index < text.size() && IsLetterByte(static_cast<unsigned char>(text[index]))) {
        ++index;
      }
    } else if (std::isdigit(item)) {
      ++index;
    } else {
      while (index < text.size() &&
             !std::isspace(static_cast<unsigned char>(text[index])) &&
             !IsLetterByte(static_cast<unsigned char>(text[index])) &&
             !std::isdigit(static_cast<unsigned char>(text[index]))) {
        ++index;
      }
      while (index < text.size() && (text[index] == '\r' || text[index] == '\n')) ++index;
    }
    pieces.emplace_back(text.substr(start, index - start));
  }
  return pieces;
}

/**
 * @brief Build an unambiguous lookup key for an adjacent BPE symbol pair.
 * @param left Left BPE symbol.
 * @param right Right BPE symbol.
 * @return Null-delimited pair key.
 */
std::string PairKey(const std::string& left, const std::string& right) {
  return left + '\0' + right;
}

}  // namespace

struct Tokenizer::Impl {
  std::unordered_map<std::string, int32_t> vocab;
  std::unordered_map<std::string, int32_t> added;
  std::vector<std::string> token_text;
  std::vector<bool> is_added;
  std::unordered_map<std::string, int32_t> merge_rank;
  std::vector<std::string> byte_encoder;
  std::unordered_map<uint32_t, uint8_t> byte_decoder;

  /**
   * @brief Merge one pretokenized piece according to loaded BPE ranks.
   * @param piece Pretokenized UTF-8/byte segment.
   * @return Final vocabulary symbols.
   */
  std::vector<std::string> EncodePiece(const std::string& piece) const {
    std::string encoded;
    for (unsigned char byte : piece) encoded += byte_encoder[byte];
    std::vector<std::string> symbols = SplitUtf8(encoded);
    while (symbols.size() > 1) {
      int32_t best_rank = std::numeric_limits<int32_t>::max();
      std::string best_pair;
      for (size_t index = 0; index + 1 < symbols.size(); ++index) {
        const std::string key = PairKey(symbols[index], symbols[index + 1]);
        const auto found = merge_rank.find(key);
        if (found != merge_rank.end() && found->second < best_rank) {
          best_rank = found->second;
          best_pair = key;
        }
      }
      if (best_pair.empty()) break;
      std::vector<std::string> merged;
      for (size_t index = 0; index < symbols.size();) {
        if (index + 1 < symbols.size() &&
            PairKey(symbols[index], symbols[index + 1]) == best_pair) {
          merged.push_back(symbols[index] + symbols[index + 1]);
          index += 2;
        } else {
          merged.push_back(symbols[index++]);
        }
      }
      symbols = std::move(merged);
    }
    return symbols;
  }
};

Tokenizer::Tokenizer() : impl_(std::make_unique<Impl>()) {}
Tokenizer::~Tokenizer() = default;
Tokenizer::Tokenizer(Tokenizer&&) noexcept = default;
Tokenizer& Tokenizer::operator=(Tokenizer&&) noexcept = default;

void Tokenizer::Load(const std::string& directory) {
  impl_->vocab = ParseStringIntMap(directory + "/vocab.json");
  impl_->added = ParseStringIntMap(directory + "/added_tokens.json");

  int32_t maximum = -1;
  for (const auto& item : impl_->vocab) maximum = std::max(maximum, item.second);
  for (const auto& item : impl_->added) maximum = std::max(maximum, item.second);
  impl_->token_text.assign(static_cast<size_t>(maximum + 1), {});
  impl_->is_added.assign(static_cast<size_t>(maximum + 1), false);
  for (const auto& item : impl_->vocab) impl_->token_text[item.second] = item.first;
  for (const auto& item : impl_->added) {
    impl_->token_text[item.second] = item.first;
    impl_->is_added[item.second] = true;
  }

  std::ifstream merges(directory + "/merges.txt");
  if (!merges) throw std::runtime_error("cannot open tokenizer merges.txt");
  std::string line;
  int32_t rank = 0;
  while (std::getline(merges, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    if (line.empty() || line.front() == '#') continue;
    const size_t separator = line.find(' ');
    if (separator == std::string::npos) continue;
    impl_->merge_rank.emplace(PairKey(line.substr(0, separator),
                                      line.substr(separator + 1)),
                              rank++);
  }

  std::vector<uint32_t> bytes;
  for (uint32_t value = 33; value <= 126; ++value) bytes.push_back(value);
  for (uint32_t value = 161; value <= 172; ++value) bytes.push_back(value);
  for (uint32_t value = 174; value <= 255; ++value) bytes.push_back(value);
  std::vector<uint32_t> codes = bytes;
  uint32_t extra = 0;
  for (uint32_t byte = 0; byte <= 255; ++byte) {
    if (std::find(bytes.begin(), bytes.end(), byte) == bytes.end()) {
      bytes.push_back(byte);
      codes.push_back(256U + extra++);
    }
  }
  impl_->byte_encoder.assign(256, {});
  for (size_t index = 0; index < bytes.size(); ++index) {
    AppendUtf8(codes[index], &impl_->byte_encoder[bytes[index]]);
    impl_->byte_decoder.emplace(codes[index], static_cast<uint8_t>(bytes[index]));
  }
}

std::vector<int32_t> Tokenizer::Encode(const std::string& text) const {
  if (impl_->vocab.empty()) throw std::logic_error("tokenizer is not loaded");
  std::vector<int32_t> result;
  std::string normal;
  const auto flush = [&]() {
    for (const std::string& piece : Pretokenize(normal)) {
      for (const std::string& token : impl_->EncodePiece(piece)) {
        const auto found = impl_->vocab.find(token);
        if (found == impl_->vocab.end()) {
          throw std::runtime_error("tokenizer vocabulary does not contain a BPE token");
        }
        result.push_back(found->second);
      }
    }
    normal.clear();
  };

  for (size_t index = 0; index < text.size();) {
    bool matched = false;
    if (text[index] == '<') {
      const size_t end = text.find('>', index + 1);
      if (end != std::string::npos) {
        const std::string candidate = text.substr(index, end - index + 1);
        const auto found = impl_->added.find(candidate);
        if (found != impl_->added.end()) {
          flush();
          result.push_back(found->second);
          index = end + 1;
          matched = true;
        }
      }
    }
    if (!matched) normal.push_back(text[index++]);
  }
  flush();
  return result;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& tokens) const {
  std::string output;
  std::string encoded;
  const auto flush = [&]() {
    for (uint32_t codepoint : Utf8Codepoints(encoded)) {
      const auto found = impl_->byte_decoder.find(codepoint);
      if (found == impl_->byte_decoder.end()) {
        throw std::runtime_error("tokenizer output contains an invalid byte token");
      }
      output.push_back(static_cast<char>(found->second));
    }
    encoded.clear();
  };
  for (int32_t token : tokens) {
    if (token < 0 || static_cast<size_t>(token) >= impl_->token_text.size() ||
        impl_->token_text[token].empty()) {
      continue;
    }
    if (impl_->is_added[token]) {
      flush();
      output += impl_->token_text[token];
    } else {
      encoded += impl_->token_text[token];
    }
  }
  flush();
  return output;
}

int32_t Tokenizer::TokenId(const std::string& token) const {
  const auto added = impl_->added.find(token);
  if (added != impl_->added.end()) return added->second;
  const auto vocab = impl_->vocab.find(token);
  if (vocab != impl_->vocab.end()) return vocab->second;
  return -1;
}

}  // namespace locateanything

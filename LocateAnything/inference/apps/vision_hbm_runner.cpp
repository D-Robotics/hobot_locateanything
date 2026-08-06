// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "hbm_session.hpp"

namespace rt = locateanything_runtime;

namespace {

constexpr int32_t kFp16 = 4;
const std::vector<int32_t> kInputShape{1, 2304, 588};
const std::vector<int32_t> kOutputShape{1, 576, 2048};

void Usage(const char *program) {
  std::fprintf(stderr,
               "usage: %s --model VISION.hbm --input vision_input.f16.bin "
               "--output vision_output.f16.bin [--backend-mask MASK]\n"
               "       %s --model VISION.hbm --server [--backend-mask MASK]\n",
               program,
               program);
}

bool ParseArgs(int argc, char **argv, std::string *model,
               std::string *input, std::string *output, bool *server,
               uint32_t *backend_mask) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--server") {
      *server = true;
    } else if (index + 1 >= argc) {
      return false;
    } else if (argument == "--model") {
      *model = argv[++index];
    } else if (argument == "--input") {
      *input = argv[++index];
    } else if (argument == "--output") {
      *output = argv[++index];
    } else if (argument == "--backend-mask") {
      try {
        const unsigned long value = std::stoul(argv[++index], nullptr, 0);
        if (value > 0xFFFFFFFFUL) return false;
        *backend_mask = static_cast<uint32_t>(value);
      } catch (...) {
        return false;
      }
    } else {
      return false;
    }
  }
  return !model->empty() && (*server || (!input->empty() && !output->empty()));
}

int64_t ElementCount(const std::vector<int32_t> &shape) {
  return std::accumulate(shape.begin(), shape.end(), int64_t{1},
                         std::multiplies<int64_t>());
}

std::string ShapeString(const std::vector<int32_t> &shape) {
  std::string value = "[";
  for (size_t index = 0; index < shape.size(); ++index) {
    if (index != 0) value += ",";
    value += std::to_string(shape[index]);
  }
  return value + "]";
}

bool ReadFile(const std::string &path, size_t expected_bytes,
              std::vector<uint8_t> *data) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) return false;
  const auto size = stream.tellg();
  if (size < 0 || static_cast<size_t>(size) != expected_bytes) return false;
  data->resize(expected_bytes);
  stream.seekg(0);
  return static_cast<bool>(stream.read(
      reinterpret_cast<char *>(data->data()),
      static_cast<std::streamsize>(expected_bytes)));
}

bool WriteFile(const std::string &path, const std::vector<uint8_t> &data) {
  const std::string temporary = path + ".tmp";
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream.write(reinterpret_cast<const char *>(data.data()),
                      static_cast<std::streamsize>(data.size()))) {
      std::remove(temporary.c_str());
      return false;
    }
  }
  if (std::rename(temporary.c_str(), path.c_str()) != 0) {
    std::remove(temporary.c_str());
    return false;
  }
  return true;
}

std::vector<std::string> SplitTabs(const std::string &value) {
  std::vector<std::string> fields;
  size_t start = 0;
  while (true) {
    const size_t separator = value.find('\t', start);
    fields.push_back(value.substr(start, separator - start));
    if (separator == std::string::npos) return fields;
    start = separator + 1;
  }
}

std::string ProtocolText(std::string value) {
  for (char &item : value) {
    if (item == '\t' || item == '\r' || item == '\n') item = ' ';
  }
  return value;
}

struct InferenceResult {
  int code = 0;
  std::string message;
  double elapsed_ms = 0.0;
  size_t output_bytes = 0;
};

InferenceResult RunOne(rt::HbmSession *session, const std::string &input_path,
                       const std::string &output_path) {
  rt::Tensor input;
  input.shape = kInputShape;
  input.dtype = kFp16;
  const size_t input_bytes = static_cast<size_t>(ElementCount(kInputShape) * 2);
  if (!ReadFile(input_path, input_bytes, &input.data)) {
    return {5, "input must contain exactly " + std::to_string(input_bytes) +
                   " FP16 bytes: " + input_path};
  }

  std::vector<rt::Tensor> outputs;
  const auto started = std::chrono::steady_clock::now();
  rt::Result result = session->ExecuteGraphByName("visual", {input}, &outputs);
  const double elapsed = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  if (!result.ok()) return {6, result.message};
  if (outputs.size() != 1 || outputs[0].shape != kOutputShape ||
      outputs[0].dtype != kFp16) {
    return {7, "unexpected visual output contract"};
  }
  if (!WriteFile(output_path, outputs[0].data)) {
    return {8, "cannot write output: " + output_path};
  }
  return {0, "", elapsed, outputs[0].data.size()};
}

}  // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string input_path;
  std::string output_path;
  bool server = false;
  uint32_t backend_mask = 0;
  if (!ParseArgs(argc, argv, &model_path, &input_path, &output_path, &server,
                 &backend_mask)) {
    Usage(argv[0]);
    return 1;
  }

  rt::HbmSession session;
  session.SetBackendMask(backend_mask);
  rt::Result result = session.Load(model_path);
  if (!result.ok()) {
    std::fprintf(stderr, "[FAIL] load: code=%d message=%s\n",
                 result.code, result.message.c_str());
    return 2;
  }
  rt::Graph *graph = session.GetGraph("visual");
  if (graph == nullptr) {
    std::fprintf(stderr, "[FAIL] graph 'visual' not found\n");
    return 3;
  }
  if (graph->GetInputShapes().size() != 1 ||
      graph->GetInputDtypes().size() != 1 ||
      graph->GetInputShapes()[0] != kInputShape ||
      graph->GetInputDtypes()[0] != kFp16) {
    std::fprintf(stderr, "[FAIL] unexpected visual input contract\n");
    return 4;
  }

  if (server) {
    std::printf("LAHBM/1\tREADY\tvisual\n");
    std::fflush(stdout);
    std::string request;
    while (std::getline(std::cin, request)) {
      if (request == "LAHBM/1\tQUIT") break;
      const std::vector<std::string> fields = SplitTabs(request);
      if (fields.size() != 5 || fields[0] != "LAHBM/1" ||
          fields[1] != "RUN" || fields[2].empty()) {
        std::printf("LAHBM/1\tERROR\t0\t1\tinvalid request frame\n");
        std::fflush(stdout);
        continue;
      }
      const InferenceResult inference = RunOne(
          &session, fields[3], fields[4]);
      if (inference.code != 0) {
        const std::string message = ProtocolText(inference.message);
        std::printf("LAHBM/1\tERROR\t%s\t%d\t%s\n", fields[2].c_str(),
                    inference.code, message.c_str());
        std::fflush(stdout);
        return inference.code;
      } else {
        std::printf("LAHBM/1\tRESULT\t%s\t%.3f\t%zu\n", fields[2].c_str(),
                    inference.elapsed_ms, inference.output_bytes);
      }
      std::fflush(stdout);
    }
    return 0;
  }

  const InferenceResult inference = RunOne(&session, input_path, output_path);
  if (inference.code != 0) {
    std::fprintf(stderr, "[FAIL] inference: code=%d message=%s\n",
                 inference.code, inference.message.c_str());
    return inference.code;
  }

  std::printf("[PASS] graph=visual input_shape=%s input_dtype=F16 "
              "output_shape=%s output_dtype=F16 output_bytes=%zu "
              "inference_ms=%.3f\n",
              ShapeString(kInputShape).c_str(), ShapeString(kOutputShape).c_str(),
              inference.output_bytes, inference.elapsed_ms);
  std::printf("[PASS] output=%s\n", output_path.c_str());
  return 0;
}

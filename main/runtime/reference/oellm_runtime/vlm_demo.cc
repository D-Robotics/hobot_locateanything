// Copyright (c) [2025] [Horizon Robotics][Horizon Bole].
//
// You can use this software according to the terms and conditions of
// the Apache v2.0.
// You may obtain a copy of Apache v2.0. at:
//
//     http: //www.apache.org/licenses/LICENSE-2.0
//
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF
// ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
// NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
// See Apache v2.0 for more details.

#include <stdio.h>
#include <unistd.h>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "nlohmann/json.hpp"
#include "xlm.h"

const char* D_ROBOTICS_ASCII_LOGO = R"(
  ██████╗       ██████╗  ██████╗ ██████╗  ██████╗ ████████╗██╗ ██████╗███████╗
  ██╔══██╗      ██╔══██╗██╔═══██╗██╔══██╗██╔═══██╗╚══██╔══╝██║██╔════╝██╔════╝
  ██║  ██║█████╗██████╔╝██║   ██║██████╔╝██║   ██║   ██║   ██║██║     ███████╗
  ██║  ██║╚════╝██╔══██╗██║   ██║██╔══██╗██║   ██║   ██║   ██║██║     ╚════██║
  ██████╔╝      ██║  ██║╚██████╔╝██████╔╝╚██████╔╝   ██║   ██║╚██████╗███████║
  ╚═════╝       ╚═╝  ╚═╝ ╚═════╝ ╚═════╝  ╚═════╝    ╚═╝   ╚═╝ ╚═════╝╚══════╝
  )";

std::string get_performance_result(xlm_model_performance_t const& performance,
                                   int32_t const request_id) {
  std::string performance_result;
  if (performance.vit_infer_cost > 0) {
    performance_result +=
        "\033[38;2;0;128;255m===== vit cost: " +
        std::to_string(performance.vit_cost) +
        " ms, vit infer cost: " + std::to_string(performance.vit_infer_cost) +
        " ms =====\033[0m\n";
  }
  if (performance.prefill_token_num > 0) {
    performance_result +=
        "\033[38;2;0;128;255m===== prefill token num: " +
        std::to_string(performance.prefill_token_num) +
        " prefill cost: " + std::to_string(performance.ttft) +
        " ms, prefill speed: " + std::to_string(performance.prefill_tps) +
        " tokens/s =====\n===== decode token num: " +
        std::to_string(performance.decode_token_num) +
        " cost per token: " + std::to_string(performance.tpot) +
        " ms, decode speed: " + std::to_string(performance.decode_tps) +
        " tokens/s =====\n";
  }

  return performance_result + "\033[0m\n";
}

void callback(xlm_result_s* result, xlm_state_e state, void* userdata) {
  if (state == XLM_STATE_ERROR) {
    std::cout << "run error" << std::endl;
  } else if (state == XLM_STATE_END) {
    std::cout << result->text << std::endl << std::endl;
    std::cout << get_performance_result(result->performance, result->request_id)
              << std::flush;
    std::cout << "[User] <<< " << std::flush;
  } else if (state == XLM_STATE_START) {
    std::cout << "\n[Assistant] >>> " << result->text << std::flush;
  } else {
    std::cout << result->text << std::flush;
  }
}

static void print_usage(int argc, char** argv) {
  std::cout << "Usage:\n"
            << "  " << argv[0] << " --config_path <config_path>"
            << " [options]\n\n"

            << "Options:\n"
            << "  -c, --config_path <config_path>   "
            << "Path to the vlm config file (required)\n"
            << "  -i, --image_path <image_path>     "
            << "Path to the local image file\n"
            << "  -h, --help                        "
            << "Show this help message\n\n"

            << "Examples:\n"
            << "  " << argv[0] << " --config_path ./qwen3vl_4b_config.json\n";
}

void show_tips() {
  std::cout << "可用指令:" << std::endl;
  std::cout << "- 输入文本：<prompt>" << std::endl;
  std::cout << "- 加载图片：/image <image_path>" << std::endl;
  std::cout << "- 重新生成：regen" << std::endl;
  std::cout << "- 清除缓存：reset" << std::endl;
  std::cout << "- 退出程序：exit\n" << std::endl;
  std::cout << "[User] <<< " << std::flush;
}

void show_greeting(std::string image_path, std::string& config_path) {
  std::cout << D_ROBOTICS_ASCII_LOGO << std::endl;

  std::ifstream config_file(config_path);
  nlohmann::json config;
  config_file >> config;

  auto extract_path = [](const std::string& raw_path) {
    std::string path = raw_path;
    size_t pos = path.rfind('/');
    if (pos != std::string::npos) {
      path = path.substr(pos + 1);
    }
    return path;
  };

  std::cout << "模型类别：图像、文本对话" << std::endl;
  std::cout << "语言模型："
            << extract_path(config["llm_model_file"].get<std::string>())
            << std::endl;
  std::cout << "视觉模型："
            << extract_path(config["vit_model_file"].get<std::string>())
            << std::endl;
  if (!image_path.empty()) {
    std::cout << "加载图像：" << extract_path(image_path) << std::endl;
  }

  std::cout << std::endl;
  show_tips();
}

size_t valid_utf8_length(const std::string& str) {
  size_t i = 0;
  const size_t len = str.size();

  while (i < len) {
    const unsigned char c = static_cast<unsigned char>(str[i]);
    size_t char_len = 0;

    if (c <= 0x7F)
      char_len = 1;
    else if ((c & 0xE0) == 0xC0)
      char_len = 2;
    else if ((c & 0xF0) == 0xE0)
      char_len = 3;
    else if ((c & 0xF8) == 0xF0)
      char_len = 4;
    else
      break;

    if (i + char_len > len) break;

    bool valid_tail =
        std::all_of(str.begin() + i + 1, str.begin() + i + char_len,
                    [](unsigned char b) { return (b & 0xC0) == 0x80; });

    if (!valid_tail) break;
    i += char_len;
  }
  return i;
}

auto read_image_path(std::string input_str,
                     std::vector<std::string>& image_paths) {
  std::istringstream iss(input_str);

  std::string path;
  while (std::getline(iss, path, ',')) {
    if (!path.empty()) {
      image_paths.push_back(path);
    }
  }

  int image_num = image_paths.size();
  std::vector<xlm_input_image_t> images(image_num);
  for (int i = 0; i < image_num; ++i) {
    images[i].image_path = image_paths[i].c_str();
  }

  return images;
}

int main(int argc, char** argv) {
  std::string image_path;
  std::string config_path;

  // Parse command arguments
  for (int i = 1; i < argc; i++) {
    try {
      std::string arg = argv[i];
      if (arg == "--help" || arg == "-h") {
        print_usage(argc, argv);
        return 1;
      } else if (arg == "--config" || arg == "-c") {
        if (i + 1 < argc) {
          config_path = argv[++i];
          std::cout << "config_path: " << config_path << std::endl;
        } else {
          print_usage(argc, argv);
          return 1;
        }
      } else if (arg == "--image_path" || arg == "-i") {
        if (i + 1 < argc) {
          image_path = argv[++i];
          std::cout << "image_path: " << image_path << std::endl;
        } else {
          print_usage(argc, argv);
          return 1;
        }
      } else {
        std::cerr << "unknown option: " << arg << std::endl;
        print_usage(argc, argv);
        return 1;
      }
    } catch (std::exception& e) {
      std::cerr << "error: " << e.what() << std::endl;
      print_usage(argc, argv);
      return 1;
    }
  }

  // Get default parameters and set custom parameters
  xlm_common_params_t param = xlm_create_default_param();
  param.config_path = config_path.c_str();
  param.model_type = XLM_MODEL_TYPE_VLM;

  // Initialize handle and register callback function
  // model inference results will be returned via the callback
  xlm_handle_t oellm_handle = nullptr;
  int ret = xlm_init(&param, callback, &oellm_handle);
  if (ret == 0) {
    std::cout << "\033[42;37;1m SUCCESS \033[0m "
              << "\033[32mVLM Demo XLM Engine is ready.\033[0m" << std::endl;
  } else {
    std::cout << "\033[41;37;1m  FAILED  \033[0m "
              << "\033[31mVLM Demo XLM Engine initialization failed.\033[0m"
              << std::endl;
    return 1;
  }

  // Construct input parameters
  // Construct request
  xlm_lm_request_t request;
  memset(&request, 0, sizeof(xlm_lm_request_t));
  request.new_chat = true;
  request.type = XLM_INPUT_MULTI_MODAL;

  // input
  std::string input_str;
  std::string last_prompt;
  std::vector<xlm_input_image_t> images;
  std::vector<std::string> image_paths;

  // init image
  if (!image_path.empty()) {
    images = read_image_path(image_path, image_paths);
  }

  show_greeting(image_path, config_path);

  while (true) {
    std::getline(std::cin, input_str);
    size_t valid_len = valid_utf8_length(input_str);
    if (valid_len != input_str.size()) {
      input_str = input_str.substr(0, valid_len);
    }
    if (input_str == "exit") {
      xlm_destroy(&oellm_handle);
      return 0;
    } else if (input_str == "regen") {
      input_str = last_prompt;
    } else if (input_str == "reset") {
      request.new_chat = true;
      std::cout << "[User] <<< " << std::flush;
      continue;
    }

    // read image
    if (input_str.rfind("/image", 0) == 0) {
      image_paths.clear();
      images = read_image_path(input_str.substr(7), image_paths);
      if (images.empty()) {
        continue;
      }
      request.multi_modal_requset.has_prompt = false;
      request.multi_modal_requset.prompt = "";
    } else {
      last_prompt = input_str;
      request.multi_modal_requset.has_prompt = true;
      request.multi_modal_requset.prompt = input_str.c_str();
    }
    request.multi_modal_requset.images = images.data();
    request.multi_modal_requset.image_num = images.size();
    request.new_chat = request.new_chat || !images.empty();

    // Construct input parameters
    xlm_input_t input;
    memset(&input, 0, sizeof(xlm_input_t));
    input.request_num = 1;
    input.requests = &request;

    // Call the inference API
    xlm_infer(oellm_handle, &input, NULL);

    // Reset input state
    request.new_chat = false;
    images.clear();
  }

  // At the end of the program, call the destroy interface to release memory
  // this interface will wait for model inference to finish before exiting
  xlm_destroy(&oellm_handle);

  return 0;
}

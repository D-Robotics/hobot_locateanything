// Copyright (c) 2026 LiuAnclouds / Kangjie Xu / D-Robotics
//
// LocateAnything S600 推理 demo (最小版)。
// Vendored from oellm_runtime/examples/vlm_demo/vlm_demo.cc with
// modifications: 砍掉交互循环, 只做 1 次 xlm_init + 1 次 xlm_infer, 先验证
// libxlm 能否加载 LA 的 3 个 hbm + tokenizer. 出 text 就算通, 坐标 token
// 解析后续加.
//
// Build:  cd main/runtime && mkdir -p build && cd build && cmake .. && make
// Run:    LD_LIBRARY_PATH=../../oellm_runtime/lib HB_DNN_USER_DEFINED_L2M_SIZES=6:6:6:6 \
//         ./locateanything_demo -c <config.json> -i <image.jpg>

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "xlm.h"

static void callback(xlm_result_s *result, xlm_state_e state, void *userdata) {
  if (state == XLM_STATE_ERROR) {
    std::cerr << "[callback] run error" << std::endl;
  } else if (state == XLM_STATE_END) {
    std::cout << "[callback] END: " << (result->text ? result->text : "")
              << std::endl;
    const auto &p = result->performance;
    std::cout << "[perf] vit_infer=" << p.vit_infer_cost << "ms"
              << " prefill_tokens=" << p.prefill_token_num
              << " prefill_tps=" << p.prefill_tps
              << " decode_tokens=" << p.decode_token_num
              << " decode_tps=" << p.decode_tps
              << " ttft=" << p.ttft << "ms tpot=" << p.tpot << "ms"
              << " e2e=" << p.end_to_end_cost << "ms" << std::endl;
  } else if (state == XLM_STATE_START) {
    std::cout << "[callback] >>> " << (result->text ? result->text : "")
              << std::flush;
  } else {
    std::cout << (result->text ? result->text : "") << std::flush;
  }
}

static void print_usage(const char *prog) {
  std::cerr << "usage: " << prog
            << " -c <config.json> -i <image.jpg> [-p \"<prompt>\"]\n";
}

int main(int argc, char **argv) {
  std::string config_path;
  std::string image_path;
  std::string prompt = "Please locate the cat and output the bounding box.";
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    if ((a == "-c" || a == "--config") && i + 1 < argc) config_path = argv[++i];
    else if ((a == "-i" || a == "--image") && i + 1 < argc) image_path = argv[++i];
    else if ((a == "-p" || a == "--prompt") && i + 1 < argc) prompt = argv[++i];
    else { print_usage(argv[0]); return 1; }
  }
  if (config_path.empty() || image_path.empty()) {
    print_usage(argv[0]);
    return 1;
  }
  std::cout << "[demo] config=" << config_path << " image=" << image_path
            << " prompt=\"" << prompt << "\"" << std::endl;

  xlm_common_params_t param = xlm_create_default_param();
  param.config_path = config_path.c_str();
  param.model_type = XLM_MODEL_TYPE_VLM;  // 先用通用 VLM, 看 libxlm 怎么判

  xlm_handle_t handle = nullptr;
  int ret = xlm_init(&param, callback, &handle);
  if (ret != 0) {
    std::cerr << "[FAIL] xlm_init ret=" << ret << std::endl;
    return 2;
  }
  std::cout << "[ok] xlm_init" << std::endl;

  xlm_lm_request_t req;
  std::memset(&req, 0, sizeof(req));
  req.new_chat = true;
  req.type = XLM_INPUT_MULTI_MODAL;
  req.multi_modal_requset.has_prompt = true;
  req.multi_modal_requset.prompt = prompt.c_str();
  xlm_input_image_t img;
  std::memset(&img, 0, sizeof(img));
  img.image_path = image_path.c_str();
  req.multi_modal_requset.images = &img;
  req.multi_modal_requset.image_num = 1;

  xlm_input_t input;
  std::memset(&input, 0, sizeof(input));
  input.request_num = 1;
  input.requests = &req;

  ret = xlm_infer(handle, &input, nullptr);
  std::cout << "[demo] xlm_infer ret=" << ret << std::endl;

  xlm_destroy(&handle);
  return ret == 0 ? 0 : 3;
}

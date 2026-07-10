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

#ifndef LLM_ENGINE_XLM_INCLUDE_XLM_H_
#define LLM_ENGINE_XLM_INCLUDE_XLM_H_

#ifdef __cplusplus
extern "C" {  // C++ 环境下生效
#endif

#include <stddef.h>
#include <stdint.h>

typedef void *xlm_handle_t;

// sampling parameters
typedef struct common_params_sampling_s {
  int32_t top_k;  // <= 0 to use vocab size
  float top_p;    // 1.0 = disabled
  float min_p;    // 0.0 = disabled
  float temp;     // <= 0.0 to sample greedily, 0.0 to not output probabilities
  float typ_p;    // typical_p, 1.0 = disabled
  int32_t min_keep;
  int32_t penalty_last_n;
  float penalty_repeat;
  float penalty_freq;
  float penalty_present;
  bool do_sample;
} common_params_sampling_t;

// 模型类型
typedef enum xlm_model_type_e {
  XLM_MODEL_TYPE_INTERNVL = 0,  // 0: internvl
  XLM_MODEL_TYPE_DEEPSEEK = 1,  // 1: deepseek
  XLM_MODEL_TYPE_QWEN = 2,      // 2: qwen (saturnv)
  XLM_MODEL_TYPE_LLAMA = 3,     // 3: llama (not support yet)
  XLM_MODEL_TYPE_INTERNLM = 4,  // 4: internlm
  XLM_MODEL_TYPE_OMNI = 5,      // 5: omni
  XLM_MODEL_TYPE_QWEN_VL = 6,   // 6: qwen-vl (saturnv)
  XLM_MODEL_TYPE_QWEN2_5 = 7,   // 7: qwen2.5
  XLM_MODEL_TYPE_PI0 = 8,       // 8: pi0
  XLM_MODEL_TYPE_QWEN3 = 9,     // 9: qwen3
  XLM_MODEL_TYPE_WHISPER = 10,  // 10: whisper
  XLM_MODEL_TYPE_VLM = 11,      // 11: vlm
  XLM_MODEL_TYPE_SPIRIT = 12,   // 12: spirit
  XLM_MODEL_TYPE_SMOLVLA = 13,  // 13: smolvla
  XLM_MODEL_TYPE_GEMMA4 = 14    // 14: gemma4
} xlm_model_type;

typedef enum xlm_image_data_type_e {
  XLM_IMAGE_DATA_TYPE_UINT8 = 0,
  XLM_IMAGE_DATA_TYPE_FP16 = 1,
  XLM_IMAGE_DATA_TYPE_FP32 = 2
} xlm_image_data_type;

// vlm parameters
typedef struct vlm_params_s {
  const char *visual_model_path;  // vlm visual模型路径
  const char *audio_model_path;   // vlm audio模型路径
  const char *text_model_path;    // vlm text模型路径
  const char *embed_tokens;       // vlm embed_tokens路径
  bool online_mode;               // vlm是否为online模式
} vlm_params_t;

// vla parameters
typedef struct vla_params_s {
  const char *siglip_model_path;     // vla visual模型路径
  const char *paligemma_model_path;  // vla vl模型路径
  const char *action_model_path;     // vla action模型路径
  const char *norm_stats_path;
} vla_params_t;

// Xlm common parameters
typedef struct xlm_common_params_s {
  const char *model_path;             // 模型路径
  vlm_params_t vlm_param;             // vlm模型参数
  vla_params_t vla_param;             // vla模型参数
  const char *token_config_path;      // tokenizer路径
  const char *config_path;            // 配置文件路径
  bool k_cache_int8;                  // 是否使用k int8量化
  xlm_model_type model_type;          // 模型类型
  int32_t context_size;               // 上下文大小
  int32_t max_img_cnt;                // 每个request的最大图像数量
  common_params_sampling_t sampling;  // 采样参数
  char *prompt_file;                  // store the external prompt file name
  char *path_prompt_cache;  // path to file for saving/loading prompt eval state
} xlm_common_params_t;

// 输入类型
typedef enum xlm_input_type_e {
  XLM_INPUT_PROMPT = 0,       // 0: prompt输入
  XLM_INPUT_TOKEN = 1,        // 1: token id输入 (not support yet)
  XLM_INPUT_MULTI_MODAL = 2,  // 2: 多模态输入
  XLM_INPUT_VLA = 3,          // 3: VLA输入
  XLM_INPUT_AUDIO = 4,        // 4: audio输入
} xlm_input_type;

typedef enum xlm_infer_backend_e {
  XLM_INFER_BACKEND_ANY = 0,      // 0: 任意核
  XLM_INFER_BACKEND_BPU_ANY = 1,  // 1: 任意bpu核
  XLM_INFER_BACKEND_BPU_0 = 2,    // 2: bpu core 0
  XLM_INFER_BACKEND_BPU_1 = 3,    // 3: bpu core 1
  XLM_INFER_BACKEND_BPU_2 = 4,    // 4: bpu core 2
  XLM_INFER_BACKEND_BPU_3 = 5     // 5: bpu core 3
} xlm_infer_backend;

typedef enum xlm_img_preprocess_type_e {
  XLM_IMG_PREPROCESS_DYNAMIC = 0,  // 0: 动态分辨率（default）
  XLM_IMG_PREPROCESS_NONE = 1,     // 1: 无预处理
} xlm_img_preprocess_type;

// 增加Bev输入
typedef struct xlm_input_tensor_s {
  uint8_t *data;      // 原始数据指针（HBM或CPU内存）
  int32_t byte_size;  // 数据总字节数（便于检查）
} xlm_input_tensor_t;
// BEV输入信息结构（每个字段一个输入Tensor）
typedef struct xlm_amap_info_s {
  xlm_input_tensor_t arrow;                  // [1, 16], int32
  xlm_input_tensor_t passable;               // [1, 16], int32
  xlm_input_tensor_t main_action;            // [1, 1], int32
  xlm_input_tensor_t assistant_action;       // [1, 1], int32
  xlm_input_tensor_t distance;               // [1, 1], int32
  xlm_input_tensor_t step_main_action;       // [1, 1], int32
  xlm_input_tensor_t step_assistant_action;  // [1, 1], int32
} xlm_amap_info_t;
// ==========================
// BEV输入信息结构
// ==========================
typedef struct xlm_bev_info_s {
  xlm_input_tensor_t bev_feats;  // [1, 26, 200, 400], float16
  xlm_input_tensor_t map_feats;  // [1, 2, 200, 400], float16
  xlm_amap_info_t amap_info;     // 包含多个 int32 类型 tensor
} xlm_bev_info_t;

// token id输入
typedef struct xlm_input_token_s {
  int32_t *tokens;      // 输入的token ids
  int32_t tokens_size;  // token数量
  char *prompt;
} xlm_input_token_t;

// 图片输入
typedef struct xlm_input_image_s {
  const char *image_path;  // 图片路径 (与image_data 二选一)
  uint8_t *image_data;     // 图片数据 (与image_path 二选一)
  int32_t image_width;     // 图片宽度
  int32_t image_height;    // 图片高度
  int32_t image_type;      // 图片格式
  xlm_image_data_type image_data_type;       // 图片数据类型
  xlm_img_preprocess_type image_preprocess;  // 图片预处理方式
} xlm_input_image_t;

typedef struct xlm_input_multi_modal_s {
  const char *prompt;         // prompt
  int32_t image_num;          // 图片数量，支持单prompt对应多img
  xlm_input_image_t *images;  // 图片数据数组
  bool has_prompt;            // 是否有prompt
} xlm_input_multi_modal_t;

typedef struct xlm_input_vla_s {
  // tokens
  xlm_input_token_t tokens;
  // image
  int32_t image_num;  // 图片数量，支持单prompt对应多img
  xlm_input_image_t *images;
  // state
  int32_t state_size;    // 状态长度
  double *state;         // 原始状态
  float *state_preproc;  // 预处理状态

  // reset
  bool reset;  // 重置仿真
} xlm_input_vla_t;

typedef struct xlm_output_embed_s {
  union {
    uint16_t *data_fp16;
    float *data_fp32;
  };
  double *data_fp64;
  //  embed
  int32_t *embed_shape;  //  embed尺寸
  int32_t embed_dim;     //  embed维度
} xlm_output_embed_t;

typedef struct xlm_output_vla_s {
  int32_t embed_num;
  xlm_output_embed_t *embeds;
  int32_t aux_num;
  xlm_output_embed_t *aux;  // e.g. paligemma output0 (fp16)
  xlm_output_embed_t *kvcache;
} xlm_output_vla_t;

typedef enum xlm_priority_type_e {
  XLM_PRIORITY_TYPE_NORMAL = 0,  // 普通，按优先级执行不抢占
  XLM_PRIORITY_TYPE_HIGH = 1,    // 高，抢占普通优先级任务
  XLM_PRIORITY_TYPE_URGENT = 2  // 紧急，抢占高优先级和普通优先级任务
} xlm_priority_type_t;

typedef struct xlm_priority_s {
  xlm_priority_type_t type;  // 任务优先级类型
  int32_t priority;  // 当type为XLM_PRIORITY_TYPE_NORMAL时生效，取值范围 0～253
} xlm_priority_t;

typedef struct xlm_ppl_s {
  bool load_ckpt;             // 是否启用断点续测功能
  int32_t text_data_num;      // 截断文本到特定长度
  int32_t max_length;         // 每次送入模型的序列长度
  int32_t stride;             // 测试步长
  const char *testcase_name;  // 测试用例的文件名
  const char *hbm_path;       // 测试模型的路径
} xlm_ppl_t;

typedef struct xlm_lm_request_s {
  int32_t request_id;        // 请求id
  xlm_input_type type;       // 输入类型
  bool new_chat;             // 是否新对话
  const char *prompt_json;   // omni读取json作为输入
  bool need_partial_result;  // 是否需要分段解码结果
  union {
    const char *prompt;                           // prompt
    xlm_input_token_t token;                      // token id
    xlm_input_multi_modal_t multi_modal_requset;  // 多模输入
    xlm_input_vla_t vla_request;                  // vla输入
  };
  const char *system_prompt;
  const char *chat_template;
  bool enable_thinking;  // 是否启用模型思考
  xlm_infer_backend infer_backend;  // 推理时绑定的bpu核
  xlm_priority_t priority;  // 任务优先级 0 ～ 255, 0 最低 255 最高
  xlm_ppl_t *ppl;           // LLM的PPL参数
} xlm_lm_request_t;

// 推理输入
typedef struct xlm_input_s {
  int32_t request_num;         // 请求数量
  xlm_lm_request_t *requests;  // LM请求数组
} xlm_input_t;

typedef struct xlm_online_video_s {
  uint8_t *y_ptr;   // nv12的y分量地址
  uint8_t *uv_ptr;  // nv12的uv分量地址
  int32_t width;    // nv12的宽度
  int32_t height;   // nv12的高度
} xlm_online_video_t;

typedef struct xlm_online_audio_s {
  const float *data;  // 音频数据首地址
  size_t data_size;   // 音频数据长度
} xlm_online_audio_t;

typedef struct xlm_online_text_s {
  const char *system_text;  // system文本内容
  const char *user_text;    // user文本内容
} xlm_online_text_t;

typedef struct xlm_model_performance_s {
  double vit_cost;            // vit 相关处理总时间 ms
  double vit_infer_cost;      // vit infer 时间 ms
  int64_t prefill_token_num;  // prefill token数量
  double prefill_tps;         // prefill 速度 tokens/s
  int64_t decode_token_num;   // decode token数量
  double decode_tps;          // decode速度 tokens/s
  double ttft;                // time to first token 首字延迟
  double tpot;  // time per output token 每输出一个token的延迟
  double end_to_end_cost;          // end to end cost 端到端耗时
  float accept_rate;               // draft接受比例
  float avg_accept_count;          // average accepted tokens per draft cycle
  int32_t draft_total_iterations;  // draft总迭代轮数
  int32_t draft_total_generated_tokens;  // 草稿累计接受token数量
  int32_t draft_accept_hist_size;        // accepted列表有效长度
  int32_t draft_accept_hist[256];        // 每轮accepted token数
  int32_t draft_accept_dist_size;        // 分布数组有效长度
  int32_t draft_accept_dist[256];        // accepted长度分布
  double asr_rtf;                        // ASR模型的RTF指标
} xlm_model_performance_t;

// 推理结果
typedef struct xlm_result_s {
  char *text;          // 推理结果文本
  int32_t request_id;  // request_id与xlm_input_t中的request_id一一对应
  xlm_model_performance_t performance;  // 模型性能
} xlm_result_t;

// 推理状态
typedef enum xlm_state_e {
  XLM_STATE_START = 0,    // 开始
  XLM_STATE_END = 1,      // 结束
  XLM_STATE_RUNNING = 2,  // 运行中
  XLM_STATE_ERROR = 3     // 错误
} xlm_state_t;

/**
 * @brief 回调函数
 *
 * @param result 推理结果
 * @param state 推理状态
 * @param userdata 用户数据
 */
typedef void (*xlm_callback_t)(xlm_result_t *result, xlm_state_t state,
                               void *userdata);

/**
 * @brief 创建默认参数数值
 *
 * @return xlm_common_params_t 默认参数数值
 */
xlm_common_params_t xlm_create_default_param();

/**
 * @brief 初始化实例
 *
 * @param param 输入参数
 * @param callback 回调函数
 * @param llm_handle 推理句柄，输出参数，由用户声明，通过init接口赋值
 * @return int 返回值
 */
int xlm_init(xlm_common_params_t *param, xlm_callback_t callback,
             void **llm_handle);

/**
 * @brief
 * 同步推理，启动推理包括一次完整的prefil和decode，不需要暴露prefill和decode细节给用户
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param input 推理输入
 * @param userdata 用户数据
 * @return int 返回值
 */
int xlm_infer(xlm_handle_t handle, xlm_input_t *input, void *userdata);

/**
 * @brief
 * 同步推理，启动仅针对prefil的ppl计算，不需要暴露prefill细节给用户
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param input 推理输入
 * @param userdata 用户数据
 * @return int 返回值
 */
int xlm_ppl(xlm_handle_t handle, xlm_input_t *input, void *userdata);

/**
 * @brief
 * 同步推理，启动OpenCompass测试
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param input 推理输入
 * @param userdata 用户数据
 * @return int 返回值
 */
int xlm_ocp(xlm_handle_t handle, xlm_input_t *input, void *userdata);

/**
 * @brief
 * 以在线方式加载nv12视频数据（当前仅用于omni模型）
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param video_input 视频输入
 * @return int32_t 执行结果，0表示成功，其他表示失败
 */
int xlm_feed_video_online(xlm_handle_t handle, xlm_online_video_t video_input);

/**
 * @brief
 * 以在线方式加载音频数据（当前仅用于omni模型）
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param audio_input 音频输入
 * @return int32_t 执行结果，0表示成功，其他表示失败
 */
int xlm_feed_audio_online(xlm_handle_t handle, xlm_online_audio_t audio_input);

/**
 * @brief
 * 以在线方式加载文本信息（当前仅用于omni模型）
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param text_input 文本输入
 * @return int32_t 执行结果，0表示成功，其他表示失败
 */
int xlm_feed_text_online(xlm_handle_t handle, xlm_online_text_t text_input);

/**
 * @brief
 * 同步推理，启动vlm的全流程处理（当前仅用于omni模型）
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param input 推理输入
 * @param userdata 用户数据
 * @return int 返回值
 */
int xlm_vlm(xlm_handle_t handle, xlm_input_t *input, void *userdata);

/**
 * @brief
 * 同步推理，启动vla的全流程处理
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param input 推理输入
 * @param userdata 用户数据
 * @param model_type 模型类型，走不同的VLA模型分支
 * @return int 返回值
 */
int xlm_vla(xlm_handle_t handle, xlm_input_t *input, xlm_output_vla_t *output,
            void *userdata, xlm_model_type model_type);

/**
 * @brief
 * 异步处理，启动推理包括一次完整的prefil和decode，不需要暴露prefill和decode细节给用户
 *
 * @param handle 推理句柄，通过xlm_init接口获取
 * @param input 推理输入
 * @param userdata 用户数据
 * @return int 返回值
 */
int xlm_infer_async(xlm_handle_t handle, xlm_input_t *input, void *userdata);

/**
 * @brief 销毁实例
 *
 * @param handle 推理句柄，通过xlm_init接口获取，在此接口内释放
 * @return int 返回值
 */
int xlm_destroy(xlm_handle_t *handle);

#ifdef __cplusplus
}
#endif

#endif  // LLM_ENGINE_XLM_INCLUDE_XLM_H_

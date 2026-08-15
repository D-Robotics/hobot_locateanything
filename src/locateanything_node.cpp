#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <ai_msgs/msg/perception_targets.hpp>
#include <ai_msgs/msg/perf.hpp>
#include <ai_msgs/msg/point.hpp>
#include <ai_msgs/msg/roi.hpp>
#include <ai_msgs/msg/target.hpp>
#include <builtin_interfaces/msg/time.hpp>
#include <hbm_img_msgs/msg/hbm_msg1080_p.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>

#include "inference.hpp"
#include "locateanything_node.hpp"
#include "package_paths.hpp"
#include "processing/image.hpp"
#include "processing/prompt.hpp"

namespace fs = std::filesystem;

namespace locateanything {
namespace {

constexpr auto kDropWarningInterval = std::chrono::seconds(5);
constexpr auto kPromptWaitWarningInterval = std::chrono::seconds(5);
constexpr size_t kFpsWindowSize = 30;

/**
 * @brief Normalize a measured duration before placing it in a ROS message.
 * @param value Measured duration in milliseconds.
 * @return Positive finite duration, or zero for an invalid value.
 */
double SafeDurationMs(double value) {
  return std::isfinite(value) && value > 0.0 ? value : 0.0;
}

/**
 * @brief Replace control characters so one inference remains one log line.
 * @param value Text that may contain line-breaking control characters.
 * @return Sanitized log text.
 */
std::string LogText(const std::string& value) {
  std::string result = value;
  for (char& item : result) {
    if (item == '\r' || item == '\n' || item == '\t') item = ' ';
  }
  return result;
}

/**
 * @brief Join prediction labels into the compact INFO-log representation.
 * @param prediction Structured inference prediction.
 * @return Pipe-separated detection and point labels.
 */
std::string ResultLabels(const Prediction& prediction) {
  std::ostringstream stream;
  bool first = true;
  const auto append = [&stream, &first](const std::string& label) {
    if (!first) stream << " | ";
    stream << LogText(label);
    first = false;
  };
  for (const Detection& detection : prediction.detections) {
    append(detection.label);
  }
  for (const Point& point : prediction.points) append(point.label);
  return stream.str();
}

/**
 * @brief Round a floating coordinate and clamp it to an image bound.
 * @param value Source-space floating coordinate.
 * @param limit Inclusive image-axis limit.
 * @return Clamped unsigned coordinate.
 */
uint32_t RoundAndClamp(float value, uint32_t limit) {
  if (!std::isfinite(value)) return 0;
  const long rounded = std::lround(value);
  return static_cast<uint32_t>(
      std::clamp<long>(rounded, 0, static_cast<long>(limit)));
}

/**
 * @brief Clamp a floating-point coordinate to the source image extent.
 * @param value Source-space coordinate.
 * @param limit Inclusive image-axis limit.
 * @return Finite clamped coordinate.
 */
float ClampPoint(float value, float limit) {
  return std::isfinite(value) ? std::clamp(value, 0.0f, limit) : 0.0f;
}

/**
 * @brief Append one inference stage timing to the outgoing ai_msgs message.
 * @param message Destination PerceptionTargets message.
 * @param type Public stage name.
 * @param inference_started ROS timestamp corresponding to inference start.
 * @param timing Monotonic stage offsets from the shared inference core.
 */
void AppendPerf(ai_msgs::msg::PerceptionTargets* message,
                const std::string& type,
                const rclcpp::Time& inference_started,
                const StageTiming& timing) {
  const double start_ms = SafeDurationMs(timing.start_ms);
  const double end_ms = std::max(start_ms, SafeDurationMs(timing.end_ms));
  const rclcpp::Time start = inference_started +
      rclcpp::Duration::from_seconds(start_ms / 1000.0);
  const rclcpp::Time end = inference_started +
      rclcpp::Duration::from_seconds(end_ms / 1000.0);
  ai_msgs::msg::Perf perf;
  perf.type = type;
  perf.stamp_start = static_cast<builtin_interfaces::msg::Time>(start);
  perf.stamp_end = static_cast<builtin_interfaces::msg::Time>(end);
  perf.time_ms_duration = end_ms - start_ms;
  message->perfs.emplace_back(std::move(perf));
}

/**
 * @brief Render token IDs for opt-in DEBUG logging.
 * @param tokens Generated model token IDs.
 * @return Comma-separated token list.
 */
std::string TokenIdsText(const std::vector<int32_t>& tokens) {
  std::ostringstream stream;
  for (size_t index = 0; index < tokens.size(); ++index) {
    if (index != 0) stream << ',';
    stream << tokens[index];
  }
  return stream.str();
}

}  // namespace

/** Latest-frame queue item with the Prompt snapshot captured on arrival. */
struct PendingFrame {
  std_msgs::msg::Header header;
  cv::Mat image;
  std::string prompt;
};

class LocateAnythingNode : public rclcpp::Node {
 public:
  /** Declare parameters, load the shared inference core, and wire TROS topics. */
  LocateAnythingNode() : Node("hobot_locateanything") {
    const std::string input_topic =
        declare_parameter<std::string>("input_topic", "/hbmem_img");
    const bool is_shared_mem_sub =
        declare_parameter<bool>("is_shared_mem_sub", true);
    const std::string prompt_topic =
        declare_parameter<std::string>("prompt_topic", "/locateanything/prompt");
    const std::string result_topic = declare_parameter<std::string>(
        "result_topic", "/perception/locateanything");
    if (input_topic.empty() || prompt_topic.empty() || result_topic.empty()) {
      throw std::invalid_argument("input, prompt, and result topics must not be empty");
    }
    const std::string model_directory_parameter =
        declare_parameter<std::string>("model_directory", "models");
    const std::string tokenizer_directory_parameter =
        declare_parameter<std::string>("tokenizer_directory", "models/tokenizer");
    if (model_directory_parameter.empty() ||
        tokenizer_directory_parameter.empty()) {
      throw std::invalid_argument(
          "model_directory and tokenizer_directory must be explicit");
    }
    const fs::path model_directory =
        ResolveRuntimePath(model_directory_parameter);
    const fs::path tokenizer_directory =
        ResolveRuntimePath(tokenizer_directory_parameter);
    const std::string l2m_sizes =
        declare_parameter<std::string>("l2m_sizes", "6:6:6:6");
    if (l2m_sizes.empty()) {
      throw std::invalid_argument("l2m_sizes must not be empty");
    }
    if (setenv("HB_DNN_USER_DEFINED_L2M_SIZES", l2m_sizes.c_str(), 1) != 0) {
      throw std::runtime_error("cannot configure S600 BPU L2 cache");
    }

    InferenceOptions options;
    options.vision_model =
        (fs::path(model_directory) /
         declare_parameter<std::string>("vision_model", "LocateAnything-3B_vision.hbm"))
            .string();
    options.language_model =
        (fs::path(model_directory) /
         declare_parameter<std::string>("language_model", "LocateAnything-3B_language.hbm"))
            .string();
    options.embeddings =
        (fs::path(model_directory) /
         declare_parameter<std::string>("embeddings", "LocateAnything-3B_embed_tokens.bin"))
            .string();
    options.tokenizer_directory = tokenizer_directory.string();
    options.image_width = declare_parameter<int>("image_width", 672);
    options.image_height = declare_parameter<int>("image_height", 672);
    options.resize_mode =
        declare_parameter<std::string>("resize_mode", "letterbox");
    options.letterbox_fill = declare_parameter<int>("letterbox_fill", 128);
    options.generation_mode =
        declare_parameter<std::string>("generation_mode", "hybrid");
    options.max_new_tokens = declare_parameter<int>("max_new_tokens", 4096);
    options.nms_iou =
        static_cast<float>(declare_parameter<double>("nms_iou", 0.9));
    const int vision_backend_mask =
        declare_parameter<int>("vision_backend_mask", 15);
    const int language_backend_mask =
        declare_parameter<int>("language_backend_mask", 15);
    if (vision_backend_mask <= 0 || vision_backend_mask > 15 ||
        language_backend_mask <= 0 || language_backend_mask > 15) {
      throw std::invalid_argument("S600 backend masks must be in [1, 15]");
    }
    options.vision_backend_mask = static_cast<uint32_t>(vision_backend_mask);
    options.language_backend_mask = static_cast<uint32_t>(language_backend_mask);

    session_ = std::make_unique<InferenceSession>(std::move(options));
    const auto initialization_started = std::chrono::steady_clock::now();
    session_->Initialize([this](const std::string& stage) {
      RCLCPP_INFO(get_logger(), "loading %s", stage.c_str());
    });
    const double initialization_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - initialization_started).count();
    RCLCPP_INFO(get_logger(), "inference core ready in %.1f s",
                initialization_seconds);
    result_publisher_ =
        create_publisher<ai_msgs::msg::PerceptionTargets>(result_topic, 10);
    prompt_subscription_ = create_subscription<std_msgs::msg::String>(
        prompt_topic, 10,
        [this](const std_msgs::msg::String::ConstSharedPtr message) {
          OnPrompt(message);
        });
    if (is_shared_mem_sub) {
      shared_image_subscription_ =
          create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
              input_topic, rclcpp::SensorDataQoS(),
              [this](const hbm_img_msgs::msg::HbmMsg1080P& message) {
                OnSharedImage(message);
              });
    } else {
      image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
          input_topic, rclcpp::SensorDataQoS(),
          [this](const sensor_msgs::msg::Image::ConstSharedPtr message) {
            OnImage(message);
          });
    }
    worker_ = std::thread([this] { Run(); });
    RCLCPP_INFO(get_logger(),
                "ready: input=%s transport=%s prompt_topic=%s result=%s",
                input_topic.c_str(),
                is_shared_mem_sub ? "hbmem" : "sensor_msgs/Image",
                prompt_topic.c_str(), result_topic.c_str());
    last_prompt_wait_warning_ = std::chrono::steady_clock::now();
    RCLCPP_WARN(get_logger(),
                "waiting for prompt on %s; image frames are ignored until a "
                "valid prompt arrives",
                prompt_topic.c_str());
  }

  /** Stop the worker and report any frames dropped during shutdown. */
  ~LocateAnythingNode() override {
    uint64_t unreported_drops = 0;
    uint64_t total_drops = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
      unreported_drops = dropped_since_warning_;
      total_drops = dropped_frames_;
    }
    if (unreported_drops > 0) {
      RCLCPP_WARN(get_logger(),
                  "latest-frame queue dropped %llu additional frame(s) before "
                  "shutdown; total_dropped=%llu",
                  static_cast<unsigned long long>(unreported_drops),
                  static_cast<unsigned long long>(total_drops));
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  /**
   * @brief Validate a public Prompt before changing active Prompt state.
   * @param prompt Public LocateAnything task command.
   */
  static void ValidatePrompt(const std::string& prompt) {
    (void)PromptBuilder{}.Build(prompt);
  }

  /**
   * @brief Validate and activate a Prompt message for subsequently received frames.
   * @param message Incoming `/locateanything/prompt` message.
   */
  void OnPrompt(const std_msgs::msg::String::ConstSharedPtr message) {
    try {
      ValidatePrompt(message->data);
    } catch (const std::exception& error) {
      RCLCPP_WARN(get_logger(),
                  "ignoring invalid prompt; previous prompt remains active: %s",
                  error.what());
      return;
    }
    {
      std::lock_guard<std::mutex> lock(mutex_);
      prompt_ = message->data;
    }
    RCLCPP_INFO(get_logger(), "prompt updated: %s",
                LogText(message->data).c_str());
  }

  /**
   * @brief Convert an official TROS shared-memory NV12 or JPEG message and queue it.
   * @param message Incoming `hbm_img_msgs/msg/HbmMsg1080P` frame.
   */
  void OnSharedImage(
      const hbm_img_msgs::msg::HbmMsg1080P& message) {
    try {
      const auto encoding_end = std::find(
          message.encoding.begin(), message.encoding.end(), uint8_t{0});
      const std::string encoding(message.encoding.begin(), encoding_end);
      if (message.data_size > message.data.size()) {
        throw std::runtime_error(
            "shared-memory data_size exceeds message buffer");
      }
      cv::Mat image;
      if (encoding == "nv12") {
        image = Nv12ToBgr(message.data.data(), message.data_size,
                          message.width, message.height, message.step);
      } else if (encoding == "jpeg" || encoding == "jpg") {
        image = JpegToBgr(message.data.data(), message.data_size);
      } else {
        throw std::runtime_error(
            "unsupported shared-memory encoding '" + encoding +
            "'; expected nv12 or jpeg");
      }
      std_msgs::msg::Header header;
      header.stamp = message.time_stamp;
      header.frame_id = std::to_string(message.index);
      QueueFrame(header, std::move(image));
    } catch (const std::exception& error) {
      RCLCPP_WARN(get_logger(), "cannot process shared image: %s",
                  error.what());
    }
  }

  /**
   * @brief Convert a standard ROS image message to BGR and queue it.
   * @param message Incoming `sensor_msgs/msg/Image` frame.
   */
  void OnImage(const sensor_msgs::msg::Image::ConstSharedPtr message) {
    try {
      cv::Mat image;
      if (message->encoding == "nv12") {
        image = Nv12ToBgr(
            reinterpret_cast<const uint8_t*>(message->data.data()),
            message->data.size(), message->width, message->height,
            message->step);
      } else if (message->encoding == "bgr" ||
                 message->encoding == "rgb") {
        image = PackedColorToBgr(
            reinterpret_cast<const uint8_t*>(message->data.data()),
            message->data.size(), message->width, message->height,
            message->step, message->encoding == "rgb");
      } else {
        image = cv_bridge::toCvCopy(message, "bgr8")->image;
      }
      QueueFrame(message->header, std::move(image));
    } catch (const std::exception& error) {
      RCLCPP_WARN(get_logger(), "cannot process input image: %s",
                  error.what());
    }
  }

  /**
   * @brief Replace the pending frame and snapshot the active Prompt atomically.
   * @param header Source message header retained for the result.
   * @param image Converted source image transferred into the latest-frame queue.
   */
  void QueueFrame(const std_msgs::msg::Header& header, cv::Mat image) {
    if (image.empty()) {
      throw std::runtime_error("input image conversion returned empty data");
    }
    bool waiting_for_prompt = false;
    bool warn_waiting_for_prompt = false;
    uint64_t warn_drops = 0;
    uint64_t total_drops = 0;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (prompt_.empty()) {
        waiting_for_prompt = true;
        const auto now = std::chrono::steady_clock::now();
        if (now - last_prompt_wait_warning_ >= kPromptWaitWarningInterval) {
          warn_waiting_for_prompt = true;
          last_prompt_wait_warning_ = now;
        }
      } else {
        if (pending_.has_value()) {
          ++dropped_frames_;
          ++dropped_since_warning_;
          const auto now = std::chrono::steady_clock::now();
          if (now - last_drop_warning_ >= kDropWarningInterval) {
            warn_drops = dropped_since_warning_;
            total_drops = dropped_frames_;
            dropped_since_warning_ = 0;
            last_drop_warning_ = now;
          }
        }
        pending_ = PendingFrame{header, std::move(image), prompt_};
      }
    }
    if (warn_waiting_for_prompt) {
      RCLCPP_WARN(get_logger(),
                  "waiting for prompt; image frames are ignored until a valid "
                  "prompt arrives");
    }
    if (waiting_for_prompt) return;
    if (warn_drops > 0) {
      RCLCPP_WARN(get_logger(),
                  "latest-frame queue dropped %llu frame(s) in the last "
                  "interval; total_dropped=%llu",
                  static_cast<unsigned long long>(warn_drops),
                  static_cast<unsigned long long>(total_drops));
    }
    condition_.notify_one();
  }

  /** @brief Consume latest queued frames until ROS shutdown or node teardown. */
  void Run() {
    while (rclcpp::ok()) {
      PendingFrame frame;
      {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [this] { return stopping_ || pending_.has_value(); });
        if (stopping_) return;
        frame = std::move(*pending_);
        pending_.reset();
      }
      const uint32_t frame_index = FrameIndex(frame.header);
      try {
        const rclcpp::Time inference_started = now();
        InferenceOutput output =
            session_->Infer(frame.image, frame.prompt, frame_index);
        const int16_t fps = RecordOutputFps(output.metrics.total_ms);
        ai_msgs::msg::PerceptionTargets result =
            BuildResult(frame, output, fps, inference_started);
        result_publisher_->publish(result);

        const LanguageMetrics& language = output.metrics.language;
        const std::string labels = ResultLabels(output.prediction);
        RCLCPP_INFO(
            get_logger(),
            "frame_id=%s prompt=\"%s\" output=\"%s\" labels=\"%s\" "
            "boxes=%zu points=%zu "
            "fps=%d stop_reason=%s prompt_tokens=%d generated_tokens=%d "
            "pbd_calls=%d pbd_accepted_tokens=%d mode=%s "
            "preprocess_ms=%.3f vision_ms=%.3f language_ms=%.3f "
            "postprocess_ms=%.3f total_ms=%.3f",
            LogText(frame.header.frame_id).c_str(), LogText(frame.prompt).c_str(),
            LogText(output.generated_text).c_str(), labels.c_str(),
            output.prediction.detections.size(),
            output.prediction.points.size(), static_cast<int>(fps),
            output.stop_reason.c_str(), language.prompt_tokens,
            language.generated_tokens, language.pbd_calls,
            language.pbd_accepted_tokens, language.executed_mode.c_str(),
            output.metrics.preprocess_ms, output.metrics.vision_ms,
            output.metrics.language_ms, output.metrics.postprocess_ms,
            output.metrics.total_ms);
        RCLCPP_DEBUG(get_logger(), "frame_id=%s generated_token_ids=[%s]",
                     LogText(frame.header.frame_id).c_str(),
                     TokenIdsText(output.generated_token_ids).c_str());
        if (!language.fallback_reason.empty()) {
          RCLCPP_DEBUG(get_logger(), "frame_id=%s language fallback: %s",
                       LogText(frame.header.frame_id).c_str(),
                       LogText(language.fallback_reason).c_str());
        }
      } catch (const std::exception& error) {
        RCLCPP_ERROR(get_logger(), "frame_id=%s inference failed: %s",
                     LogText(frame.header.frame_id).c_str(), error.what());
      }
    }
  }

  /**
   * @brief Convert shared inference output into the public ai_msgs contract.
   * @param frame Source header, image extent, and Prompt snapshot.
   * @param output Structured shared-core inference output.
   * @param fps Measured result publication throughput.
   * @param inference_started ROS timestamp corresponding to inference start.
   * @return One PerceptionTargets message, including empty-target frames.
   */
  static ai_msgs::msg::PerceptionTargets BuildResult(
      const PendingFrame& frame, const InferenceOutput& output, int16_t fps,
      const rclcpp::Time& inference_started) {
    ai_msgs::msg::PerceptionTargets message;
    message.header = frame.header;
    message.fps = fps;

    AppendPerf(&message, "preprocess", inference_started,
               output.metrics.preprocess_timing);
    AppendPerf(&message, "vision", inference_started,
               output.metrics.vision_timing);
    AppendPerf(&message, "language", inference_started,
               output.metrics.language_timing);
    AppendPerf(&message, "postprocess", inference_started,
               output.metrics.postprocess_timing);

    const uint32_t width = static_cast<uint32_t>(frame.image.cols);
    const uint32_t height = static_cast<uint32_t>(frame.image.rows);
    for (const Detection& detection : output.prediction.detections) {
      const uint32_t left = RoundAndClamp(detection.box[0], width);
      const uint32_t top = RoundAndClamp(detection.box[1], height);
      const uint32_t right = RoundAndClamp(detection.box[2], width);
      const uint32_t bottom = RoundAndClamp(detection.box[3], height);
      const uint32_t x_min = std::min(left, right);
      const uint32_t y_min = std::min(top, bottom);
      const uint32_t x_max = std::max(left, right);
      const uint32_t y_max = std::max(top, bottom);

      ai_msgs::msg::Target target;
      target.type = detection.label;
      ai_msgs::msg::Roi roi;
      roi.type = detection.label;
      roi.confidence = -1.0f;
      roi.rect.x_offset = x_min;
      roi.rect.y_offset = y_min;
      roi.rect.width = x_max - x_min;
      roi.rect.height = y_max - y_min;
      roi.rect.do_rectify = false;
      target.rois.emplace_back(std::move(roi));
      message.targets.emplace_back(std::move(target));
    }

    for (const Point& point : output.prediction.points) {
      ai_msgs::msg::Target target;
      target.type = point.label;
      ai_msgs::msg::Point point_message;
      point_message.type = point.label;
      point_message.point.emplace_back();
      point_message.point.back().x =
          ClampPoint(point.point[0], static_cast<float>(width));
      point_message.point.back().y =
          ClampPoint(point.point[1], static_cast<float>(height));
      point_message.point.back().z = 0.0f;
      point_message.confidence.emplace_back(-1.0f);
      target.points.emplace_back(std::move(point_message));
      message.targets.emplace_back(std::move(target));
    }
    return message;
  }

  /**
   * @brief Compute output throughput from the rolling completion window.
   * @param total_ms Current frame latency used before the window has two samples.
   * @return Rounded result FPS clamped to the message field range.
   */
  int16_t RecordOutputFps(double total_ms) {
    const auto timestamp = std::chrono::steady_clock::now();
    output_timestamps_.push_back(timestamp);
    while (output_timestamps_.size() > kFpsWindowSize) {
      output_timestamps_.pop_front();
    }
    if (output_timestamps_.size() < 2) {
      const double single_frame_fps = total_ms > 0.0 ? 1000.0 / total_ms : 0.0;
      return static_cast<int16_t>(std::clamp<long>(
          std::lround(single_frame_fps), 0, std::numeric_limits<int16_t>::max()));
    }
    const double seconds = std::chrono::duration<double>(
                               output_timestamps_.back() -
                               output_timestamps_.front())
                               .count();
    if (seconds <= 0.0) return -1;
    const long fps = std::lround(
        static_cast<double>(output_timestamps_.size() - 1) / seconds);
    return static_cast<int16_t>(std::clamp<long>(
        fps, 0, std::numeric_limits<int16_t>::max()));
  }

  /**
   * @brief Extract a numeric frame index when the publisher provides one.
   * @param header Source message header.
   * @return Parsed frame index, or zero for a non-numeric frame ID.
   */
  static uint32_t FrameIndex(const std_msgs::msg::Header& header) {
    try {
      const unsigned long value = std::stoul(header.frame_id);
      if (value <= std::numeric_limits<uint32_t>::max()) {
        return static_cast<uint32_t>(value);
      }
    } catch (const std::exception&) {
    }
    return 0;
  }

  std::unique_ptr<InferenceSession> session_;
  std::string prompt_;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<PendingFrame> pending_;
  bool stopping_ = false;
  uint64_t dropped_frames_ = 0;
  uint64_t dropped_since_warning_ = 0;
  std::chrono::steady_clock::time_point last_drop_warning_ =
      std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point last_prompt_wait_warning_ =
      std::chrono::steady_clock::now();
  std::deque<std::chrono::steady_clock::time_point> output_timestamps_;
  std::thread worker_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr
      shared_image_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr prompt_subscription_;
  rclcpp::Publisher<ai_msgs::msg::PerceptionTargets>::SharedPtr
      result_publisher_;
};

/** Construct the ROS node through the small package-level entry interface. */
std::shared_ptr<rclcpp::Node> CreateLocateAnythingNode() {
  return std::make_shared<LocateAnythingNode>();
}

}  // namespace locateanything

#include <algorithm>
#include <condition_variable>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>

#if __has_include(<cv_bridge/cv_bridge.hpp>)
#include <cv_bridge/cv_bridge.hpp>
#else
#include <cv_bridge/cv_bridge.h>
#endif
#include <hbm_img_msgs/msg/hbm_msg1080_p.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include "inference.hpp"
#include "locateanything_node.hpp"

namespace fs = std::filesystem;

namespace locateanything {

cv::Mat Nv12ToBgr(const uint8_t* data, size_t data_size, uint32_t width,
                  uint32_t height, uint32_t step = 0) {
  if (data == nullptr || width == 0 || height == 0 || width % 2 != 0 ||
      height % 2 != 0) {
    throw std::runtime_error("invalid NV12 image buffer");
  }
  const size_t nv12_rows = static_cast<size_t>(height) * 3 / 2;
  uint32_t row_stride = step == 0 ? width : step;
  if (data_size % nv12_rows == 0) {
    const size_t inferred_stride = data_size / nv12_rows;
    if (inferred_stride >= width) {
      row_stride = static_cast<uint32_t>(inferred_stride);
    }
  }
  const size_t required = static_cast<size_t>(row_stride) * nv12_rows;
  if (row_stride < width ||
      data_size < required) {
    throw std::runtime_error("invalid NV12 image buffer");
  }
  cv::Mat y_plane(static_cast<int>(height), static_cast<int>(width), CV_8UC1,
                  const_cast<uint8_t*>(data), row_stride);
  cv::Mat uv_plane(static_cast<int>(height / 2), static_cast<int>(width / 2),
                   CV_8UC2,
                   const_cast<uint8_t*>(data) +
                       static_cast<size_t>(row_stride) * height,
                   row_stride);
  cv::Mat bgr;
  cv::cvtColorTwoPlane(y_plane, uv_plane, bgr, cv::COLOR_YUV2BGR_NV12);
  return bgr;
}

struct PendingFrame {
  std_msgs::msg::Header header;
  cv::Mat image;
  std::string prompt;
};

class LocateAnythingNode : public rclcpp::Node {
 public:
  LocateAnythingNode() : Node("hobot_locateanything") {
    const int feed_type = declare_parameter<int>("feed_type", 1);
    if (feed_type != 0 && feed_type != 1) {
      throw std::invalid_argument("feed_type must be 0 (local image) or 1 (topic)");
    }
    const std::string image_path =
        declare_parameter<std::string>("image", "image/test_detection.jpg");
    fs::path local_image_path;
    cv::Mat local_image;
    if (feed_type == 0) {
      local_image_path = fs::absolute(image_path);
      local_image = cv::imread(local_image_path.string(), cv::IMREAD_COLOR);
      if (local_image.empty()) {
        throw std::runtime_error("image not found or unreadable: " +
                                 local_image_path.string());
      }
    }
    const std::string input_topic =
        declare_parameter<std::string>("input_topic", "/hbmem_img");
    const bool is_shared_mem_sub =
        declare_parameter<bool>("is_shared_mem_sub", true);
    const std::string prompt_topic =
        declare_parameter<std::string>("prompt_topic", "/locateanything/prompt");
    const std::string result_topic =
        declare_parameter<std::string>("result_topic", "/locateanything/result");
    const std::string annotated_topic =
        declare_parameter<std::string>("annotated_topic", "/locateanything/annotated");
    const std::string frame_complete_topic = declare_parameter<std::string>(
        "frame_complete_topic", "/locateanything/frame_complete");
    prompt_ = declare_parameter<std::string>("default_prompt", "/detect person");

    std::string model_directory =
        declare_parameter<std::string>("model_directory", "");
    if (model_directory.empty()) {
      model_directory = "models";
    }
    std::string tokenizer_directory =
        declare_parameter<std::string>("tokenizer_directory", "");
    if (tokenizer_directory.empty()) {
      tokenizer_directory = (fs::path(model_directory) / "tokenizer").string();
    }
    output_directory_ =
        declare_parameter<std::string>("output_directory", "outputs");
    save_outputs_ = declare_parameter<bool>("save_outputs", true);
    const std::string l2m_sizes =
        declare_parameter<std::string>("l2m_sizes", "6:6:6:6");
    setenv("HB_DNN_USER_DEFINED_L2M_SIZES", l2m_sizes.c_str(), 1);

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
    options.tokenizer_directory = tokenizer_directory;
    options.temporary_directory =
        (fs::path(output_directory_) / ".runtime").string();
    options.generation_mode =
        declare_parameter<std::string>("generation_mode", "hybrid");
    options.max_new_tokens = declare_parameter<int>("max_new_tokens", 4096);
    options.nms_iou =
        static_cast<float>(declare_parameter<double>("nms_iou", 0.9));
    options.vision_backend_mask =
        static_cast<uint32_t>(declare_parameter<int>("vision_backend_mask", 15));
    options.language_backend_mask =
        static_cast<uint32_t>(declare_parameter<int>("language_backend_mask", 15));

    session_ = std::make_unique<InferenceSession>(std::move(options));
    session_->Initialize();
    result_publisher_ = create_publisher<std_msgs::msg::String>(result_topic, 10);
    annotated_publisher_ =
        create_publisher<sensor_msgs::msg::Image>(annotated_topic, rclcpp::SensorDataQoS());
    frame_complete_publisher_ =
        create_publisher<std_msgs::msg::UInt32>(frame_complete_topic, 10);
    prompt_subscription_ = create_subscription<std_msgs::msg::String>(
        prompt_topic, 10, [this](const std_msgs::msg::String::ConstSharedPtr message) {
          std::lock_guard<std::mutex> lock(mutex_);
          prompt_ = message->data;
        });
    if (feed_type == 1 && is_shared_mem_sub) {
      shared_image_subscription_ =
          create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
              input_topic, rclcpp::SensorDataQoS(),
              [this](const hbm_img_msgs::msg::HbmMsg1080P& message) {
                try {
                  const auto encoding_end = std::find(
                      message.encoding.begin(), message.encoding.end(), uint8_t{0});
                  const std::string encoding(message.encoding.begin(), encoding_end);
                  if (encoding != "nv12") {
                    throw std::runtime_error("shared-memory input must use nv12 encoding");
                  }
                  cv::Mat image = Nv12ToBgr(
                      message.data.data(), message.data_size,
                      message.width, message.height, message.step);
                  std_msgs::msg::Header header;
                  header.stamp = message.time_stamp;
                  header.frame_id = std::to_string(message.index);
                  QueueFrame(header, image);
                } catch (const std::exception& error) {
                  RCLCPP_ERROR(get_logger(), "cannot convert shared image: %s", error.what());
                }
              });
    } else if (feed_type == 1) {
      image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
          input_topic, rclcpp::SensorDataQoS(),
          [this](const sensor_msgs::msg::Image::ConstSharedPtr message) {
            try {
              cv::Mat image;
              if (message->encoding == "nv12") {
                image = Nv12ToBgr(
                    reinterpret_cast<const uint8_t*>(message->data.data()),
                    message->data.size(), message->width, message->height,
                    message->step);
              } else {
                image = cv_bridge::toCvCopy(message, "bgr8")->image;
              }
              QueueFrame(message->header, image);
            } catch (const std::exception& error) {
              RCLCPP_ERROR(get_logger(), "cannot convert input image: %s", error.what());
            }
          });
    }
    worker_ = std::thread([this] { Run(); });
    if (feed_type == 0) {
      std_msgs::msg::Header header;
      header.frame_id = "0";
      QueueFrame(header, local_image);
      RCLCPP_INFO(get_logger(), "ready: local image=%s prompt=%s",
                  local_image_path.string().c_str(), prompt_topic.c_str());
    } else {
      RCLCPP_INFO(get_logger(), "ready: input=%s transport=%s prompt=%s",
                  input_topic.c_str(),
                  is_shared_mem_sub ? "hbmem" : "sensor_msgs/Image",
                  prompt_topic.c_str());
    }
  }

  ~LocateAnythingNode() override {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      stopping_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

 private:
  void QueueFrame(const std_msgs::msg::Header& header, const cv::Mat& image) {
    if (image.empty()) throw std::runtime_error("input image conversion returned empty data");
    std::lock_guard<std::mutex> lock(mutex_);
    if (prompt_.empty()) return;
    pending_ = PendingFrame{header, image.clone(), prompt_};
    condition_.notify_one();
  }

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
        InferenceOutput output =
            session_->Infer(frame.image, frame.prompt, frame_index);
        std_msgs::msg::String result;
        result.data = output.json;
        result_publisher_->publish(result);
        annotated_publisher_->publish(
            *cv_bridge::CvImage(frame.header, "bgr8", output.annotated_image)
                 .toImageMsg());
        if (save_outputs_) Save(output, frame_index);
      } catch (const std::exception& error) {
        RCLCPP_ERROR(get_logger(), "inference failed: %s", error.what());
      }
      std_msgs::msg::UInt32 completed;
      completed.data = frame_index;
      frame_complete_publisher_->publish(completed);
    }
  }

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

  void Save(const InferenceOutput& output, uint32_t frame_index) const {
    const fs::path directory(output_directory_);
    fs::create_directories(directory);
    std::ofstream(directory / "predictions.jsonl", std::ios::app)
        << output.json << '\n';
    const fs::path frame_directory = directory / "frames";
    fs::create_directories(frame_directory);
    std::ostringstream filename;
    filename << "frame_" << std::setw(6) << std::setfill('0') << frame_index
             << ".jpg";
    cv::imwrite((frame_directory / filename.str()).string(),
                output.annotated_image);
  }

  std::unique_ptr<InferenceSession> session_;
  std::string prompt_;
  std::string output_directory_;
  bool save_outputs_ = true;
  std::mutex mutex_;
  std::condition_variable condition_;
  std::optional<PendingFrame> pending_;
  bool stopping_ = false;
  std::thread worker_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_subscription_;
  rclcpp::Subscription<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr
      shared_image_subscription_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr prompt_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_publisher_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr result_publisher_;
  rclcpp::Publisher<std_msgs::msg::UInt32>::SharedPtr
      frame_complete_publisher_;
};

std::shared_ptr<rclcpp::Node> CreateLocateAnythingNode() {
  return std::make_shared<LocateAnythingNode>();
}

}  // namespace locateanything

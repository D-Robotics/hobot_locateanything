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
#include <string>
#include <thread>

#include <ament_index_cpp/get_package_prefix.hpp>
#include <ament_index_cpp/get_package_share_directory.hpp>
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

#include "inference_session.hpp"

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
  LocateAnythingNode() : Node("locateanything") {
    const std::string image_topic =
        declare_parameter<std::string>("image_topic", "/image");
    const std::string shared_image_topic =
        declare_parameter<std::string>("shared_image_topic", "/hbmem_img");
    const bool use_shared_memory =
        declare_parameter<bool>("use_shared_memory", false);
    const std::string prompt_topic =
        declare_parameter<std::string>("prompt_topic", "/locateanything/prompt");
    const std::string result_topic =
        declare_parameter<std::string>("result_topic", "/locateanything/result");
    const std::string annotated_topic =
        declare_parameter<std::string>("annotated_topic", "/locateanything/annotated");
    const std::string frame_complete_topic = declare_parameter<std::string>(
        "frame_complete_topic", "/locateanything/frame_complete");
    prompt_ = declare_parameter<std::string>("default_prompt", "/detect person");

    const fs::path package_prefix =
        ament_index_cpp::get_package_prefix("locateanything_tros");
    const fs::path package_share =
        ament_index_cpp::get_package_share_directory("locateanything_tros");
    std::string model_directory =
        declare_parameter<std::string>("model_directory", "");
    if (model_directory.empty()) {
      model_directory = (package_share / "models/LocateAnything-3B").string();
    }
    std::string tokenizer_directory =
        declare_parameter<std::string>("tokenizer_directory", "");
    if (tokenizer_directory.empty()) {
      tokenizer_directory = (package_share / "tokenizer").string();
    }
    output_directory_ =
        declare_parameter<std::string>("output_directory", "outputs");
    save_outputs_ = declare_parameter<bool>("save_outputs", true);
    const std::string l2m_sizes =
        declare_parameter<std::string>("l2m_sizes", "6:6:6:6");
    setenv("HB_DNN_USER_DEFINED_L2M_SIZES", l2m_sizes.c_str(), 1);

    InferenceOptions options;
    options.vision_runner =
        (package_prefix / "lib/locateanything_tros/vision_hbm_runner").string();
    options.language_runner =
        (package_prefix / "lib/locateanything_tros/language_hbm_runner").string();
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
    if (use_shared_memory) {
      shared_image_subscription_ =
          create_subscription<hbm_img_msgs::msg::HbmMsg1080P>(
              shared_image_topic, rclcpp::SensorDataQoS(),
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
    } else {
      image_subscription_ = create_subscription<sensor_msgs::msg::Image>(
          image_topic, rclcpp::SensorDataQoS(),
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
    const std::string& active_image_topic =
        use_shared_memory ? shared_image_topic : image_topic;
    RCLCPP_INFO(get_logger(), "ready: image=%s prompt=%s",
                active_image_topic.c_str(), prompt_topic.c_str());
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

}  // namespace locateanything

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = std::make_shared<locateanything::LocateAnythingNode>();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("locateanything"), "%s", error.what());
  }
  rclcpp::shutdown();
  return 0;
}

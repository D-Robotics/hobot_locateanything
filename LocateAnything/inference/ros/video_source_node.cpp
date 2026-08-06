#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <hbm_img_msgs/msg/hbm_msg1080_p.hpp>
#include <opencv2/videoio.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/u_int32.hpp>

#include "nv12_conversion.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace locateanything {

class VideoSourceNode : public rclcpp::Node {
 public:
  VideoSourceNode() : Node("locateanything_video_source") {
    source_ = declare_parameter<std::string>("source", "");
    const std::string topic =
        declare_parameter<std::string>("image_topic", "/hbmem_img");
    const std::string frame_complete_topic = declare_parameter<std::string>(
        "frame_complete_topic", "/locateanything/frame_complete");
    loop_ = declare_parameter<bool>("loop", false);
    if (source_.empty()) throw std::runtime_error("source video path is required");
    if (!fs::is_regular_file(source_)) {
      throw std::runtime_error("source video does not exist: " + source_);
    }
    if (!capture_.open(source_)) {
      throw std::runtime_error("cannot open source video: " + source_);
    }

    fps_ = capture_.get(cv::CAP_PROP_FPS);
    if (!(fps_ > 0.0) || fps_ > 240.0) fps_ = 25.0;
    publisher_ = create_publisher<hbm_img_msgs::msg::HbmMsg1080P>(
        topic, rclcpp::SensorDataQoS());
    completion_subscription_ = create_subscription<std_msgs::msg::UInt32>(
        frame_complete_topic, 10,
        [this](const std_msgs::msg::UInt32::ConstSharedPtr message) {
          if (!awaiting_completion_ || message->data != frame_index_) return;
          awaiting_completion_ = false;
          PublishFrame();
        });
    started_at_ = now();
    discovery_timer_ = create_wall_timer(100ms, [this] { StartWhenReady(); });
    RCLCPP_INFO(get_logger(), "waiting for subscriber: topic=%s source=%s",
                topic.c_str(), source_.c_str());
  }

 private:
  void StartWhenReady() {
    if (publisher_->get_subscription_count() == 0 ||
        completion_subscription_->get_publisher_count() == 0) {
      if ((now() - started_at_).seconds() >= 30.0) {
        RCLCPP_ERROR(get_logger(), "no image subscriber found within 30 seconds");
        rclcpp::shutdown();
      }
      return;
    }

    discovery_timer_->cancel();
    RCLCPP_INFO(get_logger(),
                "processing every video frame: source=%.3f FPS frames=%d",
                fps_, static_cast<int>(capture_.get(cv::CAP_PROP_FRAME_COUNT)));
    PublishFrame();
  }

  void PublishFrame() {
    if (publisher_->get_subscription_count() == 0 || awaiting_completion_) return;

    cv::Mat image;
    if (!capture_.read(image)) {
      if (loop_) {
        capture_.set(cv::CAP_PROP_POS_FRAMES, 0.0);
        if (!capture_.read(image)) {
          throw std::runtime_error("cannot restart source video");
        }
      } else {
        shutdown_timer_ = create_wall_timer(1s, [] { rclcpp::shutdown(); });
        RCLCPP_INFO(get_logger(), "video completed: frames=%u", frame_index_);
        return;
      }
    }

    const int width = image.cols & ~1;
    const int height = image.rows & ~1;
    const std::vector<uint8_t> nv12 = BgrToNv12(image);
    auto loaned = publisher_->borrow_loaned_message();
    if (!loaned.is_valid()) {
      throw std::runtime_error("cannot borrow shared-memory image message");
    }
    auto& message = loaned.get();
    if (nv12.size() > message.data.size()) {
      throw std::runtime_error("video frame exceeds HbmMsg1080P capacity");
    }

    message.index = static_cast<int32_t>(++frame_index_);
    message.time_stamp = static_cast<builtin_interfaces::msg::Time>(now());
    message.height = static_cast<uint32_t>(height);
    message.width = static_cast<uint32_t>(width);
    message.data_size = static_cast<uint32_t>(nv12.size());
    message.step = static_cast<uint32_t>(width);
    std::fill(message.encoding.begin(), message.encoding.end(), uint8_t{0});
    std::copy_n("nv12", 4, message.encoding.begin());
    std::copy(nv12.begin(), nv12.end(), message.data.begin());
    publisher_->publish(std::move(loaned));
    awaiting_completion_ = true;
  }

  std::string source_;
  bool loop_ = false;
  double fps_ = 25.0;
  uint32_t frame_index_ = 0;
  bool awaiting_completion_ = false;
  cv::VideoCapture capture_;
  rclcpp::Time started_at_;
  rclcpp::Publisher<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr publisher_;
  rclcpp::Subscription<std_msgs::msg::UInt32>::SharedPtr
      completion_subscription_;
  rclcpp::TimerBase::SharedPtr discovery_timer_;
  rclcpp::TimerBase::SharedPtr shutdown_timer_;
};

}  // namespace locateanything

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<locateanything::VideoSourceNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("locateanything_video_source"), "%s",
                 error.what());
  }
  rclcpp::shutdown();
  return 0;
}

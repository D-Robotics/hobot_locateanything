#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <hbm_img_msgs/msg/hbm_msg1080_p.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>

#include "nv12_conversion.hpp"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

namespace locateanything {

class ImageSourceNode : public rclcpp::Node {
 public:
  ImageSourceNode() : Node("image_source") {
    source_ = declare_parameter<std::string>("source", "");
    const std::string topic =
        declare_parameter<std::string>("image_topic", "/hbmem_img");
    if (source_.empty()) throw std::runtime_error("source image path is required");
    if (!fs::is_regular_file(source_)) {
      throw std::runtime_error("source image does not exist: " + source_);
    }

    const cv::Mat image = cv::imread(source_, cv::IMREAD_COLOR);
    width_ = image.cols & ~1;
    height_ = image.rows & ~1;
    nv12_ = BgrToNv12(image);

    publisher_ = create_publisher<hbm_img_msgs::msg::HbmMsg1080P>(
        topic, rclcpp::SensorDataQoS());
    started_at_ = now();
    discovery_timer_ = create_wall_timer(100ms, [this] { PublishWhenReady(); });
    RCLCPP_INFO(get_logger(), "waiting for subscriber: topic=%s source=%s",
                topic.c_str(), source_.c_str());
  }

 private:
  void PublishWhenReady() {
    if (publisher_->get_subscription_count() == 0) {
      if ((now() - started_at_).seconds() >= 30.0) {
        RCLCPP_ERROR(get_logger(), "no image subscriber found within 30 seconds");
        rclcpp::shutdown();
      }
      return;
    }

    auto loaned = publisher_->borrow_loaned_message();
    if (!loaned.is_valid()) {
      RCLCPP_ERROR(get_logger(), "cannot borrow shared-memory image message");
      rclcpp::shutdown();
      return;
    }
    auto& message = loaned.get();
    if (nv12_.size() > message.data.size()) {
      RCLCPP_ERROR(get_logger(), "image buffer exceeds HbmMsg1080P capacity");
      rclcpp::shutdown();
      return;
    }

    message.index = 1;
    message.time_stamp =
        static_cast<builtin_interfaces::msg::Time>(now());
    message.height = static_cast<uint32_t>(height_);
    message.width = static_cast<uint32_t>(width_);
    message.data_size = static_cast<uint32_t>(nv12_.size());
    message.step = static_cast<uint32_t>(width_);
    std::fill(message.encoding.begin(), message.encoding.end(), uint8_t{0});
    std::copy_n("nv12", 4, message.encoding.begin());
    std::copy(nv12_.begin(), nv12_.end(), message.data.begin());
    publisher_->publish(std::move(loaned));

    discovery_timer_->cancel();
    shutdown_timer_ = create_wall_timer(1s, [] { rclcpp::shutdown(); });
    RCLCPP_INFO(get_logger(), "published image: %dx%d bytes=%zu", width_,
                height_, nv12_.size());
  }

  std::string source_;
  int width_ = 0;
  int height_ = 0;
  std::vector<uint8_t> nv12_;
  rclcpp::Time started_at_;
  rclcpp::Publisher<hbm_img_msgs::msg::HbmMsg1080P>::SharedPtr publisher_;
  rclcpp::TimerBase::SharedPtr discovery_timer_;
  rclcpp::TimerBase::SharedPtr shutdown_timer_;
};

}  // namespace locateanything

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<locateanything::ImageSourceNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("image_source"), "%s",
                 error.what());
  }
  rclcpp::shutdown();
  return 0;
}

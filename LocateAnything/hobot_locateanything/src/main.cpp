#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "locateanything_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    auto node = locateanything::CreateLocateAnythingNode();
    rclcpp::executors::MultiThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("hobot_locateanything"), "%s",
                 error.what());
  }
  rclcpp::shutdown();
  return 0;
}

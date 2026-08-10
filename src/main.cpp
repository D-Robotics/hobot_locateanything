#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "locateanything_node.hpp"

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  int exit_code = 0;
  try {
    auto node = locateanything::CreateLocateAnythingNode();
    // Prompt and image callbacks share one executor thread so their receive
    // order defines the prompt snapshot captured for each queued frame.
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("hobot_locateanything"), "%s",
                 error.what());
    exit_code = 1;
  }
  rclcpp::shutdown();
  return exit_code;
}

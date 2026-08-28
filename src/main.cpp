#include <exception>
#include <memory>

#include <rclcpp/rclcpp.hpp>

#include "locateanything_node.hpp"

/**
 * @brief Initialize ROS, run the LocateAnything node, and shut ROS down cleanly.
 * @param argc Process argument count supplied to ROS.
 * @param argv Process arguments supplied to ROS.
 * @return Zero after normal shutdown, or one after startup/runtime failure.
 */
int main(int argc, char** argv) {
  int exit_code = 0;
  try {
    rclcpp::init(argc, argv);
    auto node = locateanything::CreateLocateAnythingNode();
    // Prompt and image callbacks share one executor thread so their receive
    // order defines the prompt snapshot captured for each queued frame.
    rclcpp::executors::SingleThreadedExecutor executor;
    executor.add_node(node);
    executor.spin();
    rclcpp::shutdown();
  } catch (const std::exception& error) {
    RCLCPP_ERROR(rclcpp::get_logger("hobot_locateanything"), "%s",
                 error.what());
    exit_code = 1;
    if (rclcpp::ok()) rclcpp::shutdown();
  }
  return exit_code;
}

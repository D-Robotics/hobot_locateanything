#pragma once

#include <memory>

namespace rclcpp {
class Node;
}

namespace locateanything {

/**
 * @brief Construct the ROS node that consumes TROS images and publishes results.
 * @return Shared ROS node instance for registration with an executor.
 */
std::shared_ptr<rclcpp::Node> CreateLocateAnythingNode();

}  // namespace locateanything

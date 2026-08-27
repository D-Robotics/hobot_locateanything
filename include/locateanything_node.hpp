#pragma once

#include <memory>

namespace rclcpp {
class Node;
}

namespace locateanything {

/**
 * @brief Construct the ROS node that consumes TROS images and publishes results.
 * @return Shared ROS node instance for registration with an executor.
 * @throws std::invalid_argument if a configured inference value is invalid.
 * @throws std::runtime_error if runtime assets or HBM contracts are invalid.
 */
std::shared_ptr<rclcpp::Node> CreateLocateAnythingNode();

}  // namespace locateanything

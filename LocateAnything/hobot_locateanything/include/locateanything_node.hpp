#pragma once

#include <memory>

namespace rclcpp {
class Node;
}

namespace locateanything {

std::shared_ptr<rclcpp::Node> CreateLocateAnythingNode();

}  // namespace locateanything

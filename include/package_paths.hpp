#pragma once

#include <filesystem>

namespace locateanything {

/**
 * @brief Return the active ROS installation prefix for this package.
 * @return Prefix selected by the sourced ROS environment.
 */
std::filesystem::path PackagePrefix();

/**
 * @brief Return the directory containing installed runtime resources.
 * @return `<prefix>/lib/hobot_locateanything`.
 */
std::filesystem::path PackageRuntimeDirectory();

/**
 * @brief Resolve a configured resource against the package runtime directory.
 * @param path Absolute path or path relative to the runtime directory.
 * @return Normalized resource path.
 */
std::filesystem::path ResolveRuntimePath(const std::filesystem::path& path);

}  // namespace locateanything

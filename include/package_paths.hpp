#pragma once

#include <filesystem>

namespace locateanything {

/**
 * @brief Return the active ROS installation prefix for this package.
 * @return Prefix selected by the sourced ROS environment.
 * @throws ament_index_cpp::PackageNotFoundError if the package is not indexed.
 */
std::filesystem::path PackagePrefix();

/**
 * @brief Return the directory containing installed runtime resources.
 * @return `<prefix>/lib/hobot_locateanything`.
 * @throws ament_index_cpp::PackageNotFoundError if the package is not indexed.
 */
std::filesystem::path PackageRuntimeDirectory();

/**
 * @brief Resolve a configured resource against the package runtime directory.
 * @param[in] path Absolute path or path relative to the runtime directory.
 * @return Normalized resource path.
 * @throws ament_index_cpp::PackageNotFoundError when resolving a relative path
 * and the package is not indexed.
 */
std::filesystem::path ResolveRuntimePath(const std::filesystem::path& path);

}  // namespace locateanything

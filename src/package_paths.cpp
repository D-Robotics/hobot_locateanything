#include "package_paths.hpp"

#include <ament_index_cpp/get_package_prefix.hpp>

namespace fs = std::filesystem;

namespace locateanything {
namespace {

constexpr char kPackageName[] = "hobot_locateanything";

}  // namespace

/**
 * @brief Query ament for the active package installation prefix.
 * @return Prefix selected by the sourced ROS environment.
 */
fs::path PackagePrefix() {
  return fs::path(ament_index_cpp::get_package_prefix(kPackageName));
}

/**
 * @brief Build the installed runtime resource directory.
 * @return `<prefix>/lib/hobot_locateanything`.
 */
fs::path PackageRuntimeDirectory() {
  return PackagePrefix() / "lib" / kPackageName;
}

/**
 * @brief Resolve a configured resource against the runtime directory.
 * @param path Absolute path or path relative to the runtime directory.
 * @return Normalized resource path.
 */
fs::path ResolveRuntimePath(const fs::path& path) {
  if (path.is_absolute()) return path.lexically_normal();
  return (PackageRuntimeDirectory() / path).lexically_normal();
}

}  // namespace locateanything

#pragma once

#include <filesystem>

namespace rl::utils {

std::filesystem::path ensure_directory(const std::filesystem::path& path);

}  // namespace rl::utils

#include "utils/filesystem.hpp"

#include <stdexcept>

namespace rl::utils {

std::filesystem::path ensure_directory(const std::filesystem::path& path) {
    if (path.empty()) {
        return ".";
    }

    std::error_code error;
    std::filesystem::create_directories(path, error);
    if (error) {
        throw std::runtime_error(
            "failed to create directory: " + path.string() + ": " +
            error.message());
    }

    return path;
}

}  // namespace rl::utils

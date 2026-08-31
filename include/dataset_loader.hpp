#pragma once

#include <string>
#include <vector>
#include <memory>
#include "types.hpp"

namespace oct {

class DatasetLoader {
public:
    static std::shared_ptr<Dataset> load(
        const std::string& config_path,
        const std::string& target_uuid = ""
    );

    static std::shared_ptr<Dataset> create_synthetic(
        int num_wavelengths = 751,
        int height = 1504,
        int width = 1504
    );
};

} // namespace oct

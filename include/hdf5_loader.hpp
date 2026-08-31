#pragma once

#include <string>
#include <vector>
#include <memory>
#include "types.hpp"

namespace oct {

class Hdf5Loader {
public:
    /**
     * Loads native HDF5 SS-FF-OCT dataset from /object/images (or /reference/images).
     * Extracts first half of frames (e.g. 751 of 1502 frames) and metadata attributes.
     */
    static std::shared_ptr<Dataset> load(
        const std::string& hdf5_path,
        bool is_reference = false,
        int max_frames = 751
    );

    /**
     * Checks if a file is a valid HDF5 OCT measurement.
     */
    static bool is_valid_hdf5(const std::string& path);
};

} // namespace oct

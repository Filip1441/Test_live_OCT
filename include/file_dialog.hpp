#pragma once

#include <string>
#include <vector>

namespace oct {

class FileDialog {
public:
    /**
     * Opens a native OS File Open Dialog to pick an HDF5 measurement or reference file.
     * Returns absolute path, or empty string if cancelled.
     */
    static std::string open_file(
        const std::string& title = "Select OCT Measurement (HDF5)",
        const std::vector<std::string>& extensions = {"hdf5", "h5"}
    );
};

} // namespace oct

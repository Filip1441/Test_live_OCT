#pragma once

#include <vector>
#include <string>
#include <cstdint>

namespace oct {

enum class ColormapType {
    Inferno = 0,
    Viridis = 1,
    Magma = 2,
    Plasma = 3,
    Turbo = 4,
    Cividis = 5,
    Grayscale = 6,
    Jet = 7
};

class ColormapManager {
public:
    static const std::vector<std::string>& get_names();
    static const uint32_t* get_lut(ColormapType type);
    static const uint32_t* get_lut_by_name(const std::string& name);

    /**
     * Converts a float dB array to 32-bit RGBA pixel buffer using the chosen colormap and dynamic range.
     */
    static void apply_colormap(
        const float* __restrict__ src_db,
        size_t count,
        float db_min,
        float db_max,
        ColormapType type,
        uint32_t* __restrict__ dst_rgba
    );
};

} // namespace oct

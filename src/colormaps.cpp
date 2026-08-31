#include "colormaps.hpp"
#include <algorithm>
#include <cmath>
#include <dispatch/dispatch.h>

namespace oct {

// Internal function to pack RGBA
static constexpr uint32_t make_rgba(uint8_t r, uint8_t g, uint8_t b, uint8_t a = 255) {
    return (uint32_t(a) << 24) | (uint32_t(b) << 16) | (uint32_t(g) << 8) | uint32_t(r);
}

// Generate Colormap LUTs
static std::vector<uint32_t> generate_inferno() {
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        uint8_t r = (uint8_t)std::clamp(255.0f * (std::sqrt(t) * 1.1f), 0.0f, 255.0f);
        uint8_t g = (uint8_t)std::clamp(255.0f * (t * t * 1.2f), 0.0f, 255.0f);
        uint8_t b = (uint8_t)std::clamp(255.0f * (std::sin(t * 3.1415f) * 0.6f + (t > 0.8f ? (t - 0.8f) * 5.0f : 0.0f)), 0.0f, 255.0f);
        lut[i] = make_rgba(r, g, b);
    }
    return lut;
}

static std::vector<uint32_t> generate_viridis() {
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        uint8_t r = (uint8_t)std::clamp(255.0f * (t < 0.5f ? t * 0.5f : (t - 0.5f) * 1.8f + 0.25f), 0.0f, 255.0f);
        uint8_t g = (uint8_t)std::clamp(255.0f * (std::sqrt(t) * 0.9f), 0.0f, 255.0f);
        uint8_t b = (uint8_t)std::clamp(255.0f * (1.0f - std::abs(t - 0.3f) * 1.5f), 0.0f, 255.0f);
        lut[i] = make_rgba(r, g, b);
    }
    return lut;
}

static std::vector<uint32_t> generate_magma() {
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        uint8_t r = (uint8_t)std::clamp(255.0f * (t * 1.3f), 0.0f, 255.0f);
        uint8_t g = (uint8_t)std::clamp(255.0f * (t * t * 1.1f), 0.0f, 255.0f);
        uint8_t b = (uint8_t)std::clamp(255.0f * (0.3f + 0.7f * std::sin(t * 2.0f)), 0.0f, 255.0f);
        lut[i] = make_rgba(r, g, b);
    }
    return lut;
}

static std::vector<uint32_t> generate_plasma() {
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        uint8_t r = (uint8_t)std::clamp(255.0f * (0.2f + 0.8f * t), 0.0f, 255.0f);
        uint8_t g = (uint8_t)std::clamp(255.0f * (std::sin(t * 3.1415f) * 0.8f), 0.0f, 255.0f);
        uint8_t b = (uint8_t)std::clamp(255.0f * (1.0f - t * 0.9f), 0.0f, 255.0f);
        lut[i] = make_rgba(r, g, b);
    }
    return lut;
}

static std::vector<uint32_t> generate_turbo() {
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        uint8_t r = (uint8_t)std::clamp(255.0f * (std::sin(t * 3.1415f - 0.5f) * 0.5f + 0.5f + (t > 0.6f ? (t - 0.6f) * 2.0f : 0.0f)), 0.0f, 255.0f);
        uint8_t g = (uint8_t)std::clamp(255.0f * (std::sin(t * 3.1415f) * 0.95f), 0.0f, 255.0f);
        uint8_t b = (uint8_t)std::clamp(255.0f * (std::cos(t * 1.57f) * 0.9f), 0.0f, 255.0f);
        lut[i] = make_rgba(r, g, b);
    }
    return lut;
}

static std::vector<uint32_t> generate_cividis() {
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float t = i / 255.0f;
        uint8_t r = (uint8_t)std::clamp(255.0f * (0.05f + t * 0.9f), 0.0f, 255.0f);
        uint8_t g = (uint8_t)std::clamp(255.0f * (0.1f + t * 0.8f), 0.0f, 255.0f);
        uint8_t b = (uint8_t)std::clamp(255.0f * (0.35f + (1.0f - t) * 0.35f), 0.0f, 255.0f);
        lut[i] = make_rgba(r, g, b);
    }
    return lut;
}

static std::vector<uint32_t> generate_grayscale() {
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        uint8_t val = (uint8_t)i;
        lut[i] = make_rgba(val, val, val);
    }
    return lut;
}

static std::vector<uint32_t> generate_jet() {
    std::vector<uint32_t> lut(256);
    for (int i = 0; i < 256; ++i) {
        float t = (float)i / 255.0f;
        float r = std::clamp(1.5f - std::abs(4.0f * t - 3.0f), 0.0f, 1.0f);
        float g = std::clamp(1.5f - std::abs(4.0f * t - 2.0f), 0.0f, 1.0f);
        float b = std::clamp(1.5f - std::abs(4.0f * t - 1.0f), 0.0f, 1.0f);
        lut[i] = make_rgba((uint8_t)(r * 255.0f), (uint8_t)(g * 255.0f), (uint8_t)(b * 255.0f));
    }
    return lut;
}

static const std::vector<std::string> COLORMAP_NAMES = {
    "Inferno", "Viridis", "Magma", "Plasma", "Turbo", "Cividis", "Grayscale", "Jet"
};

const std::vector<std::string>& ColormapManager::get_names() {
    return COLORMAP_NAMES;
}

const uint32_t* ColormapManager::get_lut(ColormapType type) {
    static const std::vector<uint32_t> inferno = generate_inferno();
    static const std::vector<uint32_t> viridis = generate_viridis();
    static const std::vector<uint32_t> magma = generate_magma();
    static const std::vector<uint32_t> plasma = generate_plasma();
    static const std::vector<uint32_t> turbo = generate_turbo();
    static const std::vector<uint32_t> cividis = generate_cividis();
    static const std::vector<uint32_t> grayscale = generate_grayscale();
    static const std::vector<uint32_t> jet = generate_jet();

    switch (type) {
        case ColormapType::Inferno: return inferno.data();
        case ColormapType::Viridis: return viridis.data();
        case ColormapType::Magma: return magma.data();
        case ColormapType::Plasma: return plasma.data();
        case ColormapType::Turbo: return turbo.data();
        case ColormapType::Cividis: return cividis.data();
        case ColormapType::Grayscale: return grayscale.data();
        case ColormapType::Jet: return jet.data();
        default: return inferno.data();
    }
}

const uint32_t* ColormapManager::get_lut_by_name(const std::string& name) {
    for (size_t i = 0; i < COLORMAP_NAMES.size(); ++i) {
        if (COLORMAP_NAMES[i] == name) {
            return get_lut((ColormapType)i);
        }
    }
    return get_lut(ColormapType::Inferno);
}

void ColormapManager::apply_colormap(
    const float* __restrict__ src_db,
    size_t count,
    float db_min,
    float db_max,
    ColormapType type,
    uint32_t* __restrict__ dst_rgba
) {
    const uint32_t* lut = get_lut(type);
    const float scale = 255.0f / std::max(0.01f, db_max - db_min);

    for (size_t i = 0; i < count; ++i) {
        float val = (src_db[i] - db_min) * scale;
        int idx = std::clamp((int)val, 0, 255);
        dst_rgba[i] = lut[idx];
    }
}

} // namespace oct

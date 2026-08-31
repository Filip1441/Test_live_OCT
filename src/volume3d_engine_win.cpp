#include "volume3d_engine.hpp"
#include <iostream>
#include <algorithm>

namespace oct {

Volume3DEngine::Volume3DEngine(NativeDevicePtr device, int width, int height, int num_lambdas)
    : device_(device), width_(width), height_(height), num_lambdas_(num_lambdas) {
    ph_reconstructed_.resize((size_t)width * height, 0.0f);
}

Volume3DEngine::~Volume3DEngine() = default;

void Volume3DEngine::reinitialize(int width, int height, int num_lambdas) {
    width_ = width;
    height_ = height;
    num_lambdas_ = num_lambdas;
    ph_reconstructed_.resize((size_t)width * height, 0.0f);
}

bool Volume3DEngine::reconstruct_phase_oct(
    const float* raw_obj,
    const float* raw_ref,
    const std::vector<double>& wavelengths_nm,
    float dz_focus_um,
    float n_immersion,
    float dx_um,
    bool apply_detrend,
    PipelineMode pipeline_mode,
    ComputeBackend3D backend,
    PhaseOctStats& out_stats
) {
    std::cerr << "[Volume3DEngine] Phase-OCT reconstruction on Windows (CPU/CUDA) is currently a stub." << std::endl;
    std::cerr << "Please implement the core logic using cpu_compute_engine or cuda_compute_engine here." << std::endl;
    has_reconstructed_ = true;
    return true;
}

bool Volume3DEngine::reconstruct_3d_volume(
    const float* volume_data,
    ComputeBackend3D backend,
    Volume3DStats& out_stats
) {
    return false;
}

void Volume3DEngine::render_raymarching(
    float yaw, float pitch, float zoom, float threshold_db, float density, float z_stretch,
    int render_mode, const uint32_t* colormap_lut, int num_steps
) {
    // Stub
}

void* Volume3DEngine::get_rendered_texture_ptr() const {
    return nullptr;
}

void* Volume3DEngine::get_phase2d_texture_ptr() const {
    return nullptr;
}

void Volume3DEngine::compute_k_grid_and_dispersion(...) {}
void Volume3DEngine::find_z_layers(...) {}
void Volume3DEngine::render_phase2d_texture(const uint32_t* lut) {}
void Volume3DEngine::update_surface3d_render(float yaw, float pitch, float zoom, float z_scale, const uint32_t* lut) {}

} // namespace oct

#pragma once

#include <vector>
#include <memory>
#include <string>
#include <complex>

#ifdef __APPLE__
    #import <Metal/Metal.h>
    using NativeDevicePtr = id<MTLDevice>;
    using NativeCommandQueuePtr = id<MTLCommandQueue>;
    using NativeTexturePtr = id<MTLTexture>;
#else
    // Mock types for Windows/Linux compilation
    using NativeDevicePtr = void*;
    using NativeCommandQueuePtr = void*;
    using NativeTexturePtr = void*;
#endif

#include "types.hpp"
#include "config.hpp"
#include "colormaps.hpp"

namespace oct {

enum class ComputeBackend3D {
    CPUSingleThread = 0,
    CPUMultiCore = 1,
    MetalGPU = 2
};

enum class PipelineMode {
    Optimized_DTFT = 1,
    Matlab_Exact_FFT = 2
};

struct PhaseOctStats {
    float compute_time_ms = 0.0f;
    int z_obj_index = 0;
    int z_ref_index = 0;
    float phase_min_rad = 0.0f;
    float phase_max_rad = 0.0f;
    float step_phase_delay_rad = 0.0f;
    float step_height_um = 0.0f;
    bool is_valid = false;
};

using Volume3DStats = PhaseOctStats;

class Volume3DEngine {
public:
    Volume3DEngine(NativeDevicePtr device, int width = RAW_WIDTH, int height = RAW_HEIGHT, int num_lambdas = SWEEP_POINTS);
    ~Volume3DEngine();

    void reinitialize(int width, int height, int num_lambdas);

    // Runs the Unwrapped Phase-OCT Reconstruction Pipeline
    bool reconstruct_phase_oct(
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
    );

    // Compatibility wrapper
    bool reconstruct_3d_volume(
        const float* volume_data,
        ComputeBackend3D backend,
        Volume3DStats& out_stats
    );

    // Render interactive 3D Topography Surface
    void render_raymarching(
        float yaw,
        float pitch,
        float zoom,
        float threshold_db,
        float density,
        float z_stretch,
        int render_mode,
        const uint32_t* colormap_lut,
        int num_steps
    );

    void* get_rendered_texture_ptr() const;
    void* get_phase2d_texture_ptr() const;

    bool is_volume_reconstructed() const { return has_reconstructed_; }
    const std::vector<float>& get_phase_reconstructed() const { return ph_reconstructed_; }

private:
    void compute_k_grid_and_dispersion(
        const float* raw_ref,
        const std::vector<double>& lambda_vec,
        std::vector<float>& phi_ref,
        std::vector<float>& phi_lin,
        std::vector<std::complex<float>>& disp_filter,
        std::vector<float>& win
    );

    void find_z_layers(
        const float* raw_obj,
        const float* raw_ref,
        const std::vector<float>& phi_ref,
        const std::vector<float>& phi_lin,
        const std::vector<std::complex<float>>& disp_filter,
        const std::vector<float>& win,
        int& out_z_obj,
        int& out_z_ref
    );

    void render_phase2d_texture(const uint32_t* lut);
    void update_surface3d_render(float yaw, float pitch, float zoom, float z_scale, const uint32_t* lut);

    NativeDevicePtr device_ = nullptr;
    NativeCommandQueuePtr command_queue_ = nullptr;

    int width_ = RAW_WIDTH;
    int height_ = RAW_HEIGHT;
    int num_lambdas_ = SWEEP_POINTS;

    bool has_reconstructed_ = false;

    // Phase-OCT results
    std::vector<float> ph_reconstructed_; // [Height, Width] in radians
    std::vector<uint32_t> phase2d_rgba_;  // [Height, Width] RGBA pixels
    std::vector<uint32_t> surface3d_rgba_;// [512, 512] 3D rendered surface viewport

    NativeTexturePtr phase2d_texture_ = nullptr;
    NativeTexturePtr surface3d_texture_ = nullptr;
};

} // namespace oct

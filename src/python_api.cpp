#include <iostream>
#include <vector>
#include <memory>
#include <algorithm>
#include "volume3d_engine.hpp"

extern "C" {

#ifdef _WIN32
#define EXPORT_API __declspec(dllexport)
#else
#define EXPORT_API __attribute__((visibility("default")))
#endif

// C-API for Python ctypes
// Reconstructs phase OCT using zero-copy pointers
EXPORT_API void run_phase_oct_recon(
    const float* obj_data,
    const float* ref_data,
    int width,
    int height,
    int num_lambdas,
    float n_immersion,
    bool use_matlab_exact,
    bool show_gui,
    float* out_phase
) {
    if (!obj_data || !ref_data || !out_phase) return;

    // On Windows/Linux, we don't use Metal Device
    void* device = nullptr;
    auto engine = std::make_unique<oct::Volume3DEngine>(device, width, height, num_lambdas);
    
    oct::PhaseOctStats stats;
    oct::PipelineMode p_mode = use_matlab_exact ? oct::PipelineMode::Matlab_Exact_FFT : oct::PipelineMode::Optimized_DTFT;
    
    // Default wavelengths for 803-878nm range (251 points)
    std::vector<double> wavelengths(num_lambdas);
    double l_start = 803.0;
    double l_end = 878.0;
    for(int i=0; i<num_lambdas; ++i) {
        wavelengths[i] = l_start + (l_end - l_start) * (double)i / (double)(num_lambdas - 1);
    }

#ifdef OCT_ENABLE_CUDA
    oct::ComputeBackend3D backend = oct::ComputeBackend3D::MetalGPU; // Use CUDA mapped as GPU, though CUDA backend enum isn't explicit here yet. Wait, we should use CPUMultiCore or CUDA if implemented.
#else
    oct::ComputeBackend3D backend = oct::ComputeBackend3D::CPUMultiCore;
#endif

    engine->reconstruct_phase_oct(
        obj_data,
        ref_data,
        wavelengths,
        0.0f,
        n_immersion,
        0.5f,
        false, // no detrend
        p_mode,
        backend,
        stats
    );

    const auto& ph_recon = engine->get_phase_reconstructed();
    std::copy(ph_recon.begin(), ph_recon.end(), out_phase);

    if (show_gui) {
        std::cerr << "[Warning] C++ GUI from Python shared library is experimental and not yet implemented for Windows." << std::endl;
    }
}

} // extern "C"

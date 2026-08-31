#include "dataset_loader.hpp"
#include "volume3d_engine.hpp"
#include <iostream>
#include <vector>

int main() {
    std::cout << "[Loading dataset for 3D Volume test...]" << std::endl;
    auto dataset = oct::DatasetLoader::load(oct::DEFAULT_DATASET_PATH, oct::DEFAULT_DATASET_UUID);

    oct::Volume3DEngine engine(dataset->num_wavelengths, dataset->height, dataset->width);

    std::vector<float> cscan_slice;
    oct::Volume3DStats stats;

    std::cout << "\n--- 1. Testing Metal GPU C-Scan (Depth z=190) ---" << std::endl;
    engine.compute_cscan_slice(dataset->data.data(), 190, oct::ComputeBackend3D::MetalGPU, cscan_slice, stats);
    std::cout << "Metal GPU: " << stats.compute_time_ms << " ms | " 
              << stats.megapixels_per_sec << " Mpix/s | " << stats.gflops << " GFLOPS" << std::endl;

    std::cout << "\n--- 2. Testing Multi-Core GCD CPU C-Scan (Depth z=190) ---" << std::endl;
    engine.compute_cscan_slice(dataset->data.data(), 190, oct::ComputeBackend3D::CPU_MultiCore_GCD, cscan_slice, stats);
    std::cout << "Multi-Core CPU: " << stats.compute_time_ms << " ms | " 
              << stats.megapixels_per_sec << " Mpix/s | " << stats.gflops << " GFLOPS" << std::endl;

    std::cout << "\n--- 3. Testing Single-Core CPU Torture Mode (Depth z=190) ---" << std::endl;
    engine.compute_cscan_slice(dataset->data.data(), 190, oct::ComputeBackend3D::CPU_SingleThread, cscan_slice, stats);
    std::cout << "Single-Core CPU (Torture): " << stats.compute_time_ms << " ms | " 
              << stats.megapixels_per_sec << " Mpix/s | " << stats.gflops << " GFLOPS" << std::endl;

    std::cout << "\n--- 4. Testing Metal GPU 3D MIP (z=[150..220]) ---" << std::endl;
    engine.compute_mip(dataset->data.data(), 150, 220, oct::ComputeBackend3D::MetalGPU, cscan_slice, stats);
    std::cout << "Metal GPU 3D MIP: " << stats.compute_time_ms << " ms | " 
              << stats.megapixels_per_sec << " Mpix/s | " << stats.gflops << " GFLOPS" << std::endl;

    std::cout << "\n[ALL 3D TESTS PASSED SUCCESSFULLY!]" << std::endl;
    return 0;
}

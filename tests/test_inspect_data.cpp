#include "dataset_loader.hpp"
#include "cpu_compute_engine.hpp"
#include <iostream>
#include <cmath>

int main() {
    auto dataset = oct::DatasetLoader::load(oct::DEFAULT_DATASET_PATH, oct::DEFAULT_DATASET_UUID);
    std::cout << "Loaded dataset: num_wavelengths=" << dataset->num_wavelengths 
              << ", height=" << dataset->height << ", width=" << dataset->width << std::endl;

    const float* frame0 = dataset->get_frame(0);
    float min_val = frame0[0], max_val = frame0[0], sum = 0.0f;
    for (size_t i = 0; i < (size_t)dataset->height * dataset->width; ++i) {
        min_val = std::min(min_val, frame0[i]);
        max_val = std::max(max_val, frame0[i]);
        sum += frame0[i];
    }
    std::cout << "Frame 0 stats: min=" << min_val << ", max=" << max_val 
              << ", mean=" << (sum / (dataset->height * dataset->width)) << std::endl;

    // Test A-scan at center
    int cx = dataset->width / 2;
    int cy = dataset->height / 2;
    std::cout << "Center pixel (cx=" << cx << ", cy=" << cy << ") over all lambdas:" << std::endl;
    for (int l = 0; l < 10; ++l) {
        std::cout << "  lambda[" << l << "] = " << dataset->get_frame(l)[cy * dataset->width + cx] << std::endl;
    }

    // Now test CPU FFT on center lines
    oct::CpuComputeEngine engine(dataset->num_wavelengths);
    std::vector<float> h_raw((size_t)dataset->num_wavelengths * dataset->width);
    std::vector<float> v_raw((size_t)dataset->num_wavelengths * dataset->height);
    std::vector<float> a_raw(dataset->num_wavelengths);

    for (int l = 0; l < dataset->num_wavelengths; ++l) {
        const float* f = dataset->get_frame(l);
        for (int x = 0; x < dataset->width; ++x) {
            h_raw[l * dataset->width + x] = f[cy * dataset->width + x];
        }
        for (int y = 0; y < dataset->height; ++y) {
            v_raw[l * dataset->height + y] = f[y * dataset->width + cx];
        }
        a_raw[l] = f[cy * dataset->width + cx];
    }

    oct::CrosshairReconResult recon;
    engine.compute_from_lines(
        h_raw.data(), v_raw.data(), a_raw.data(),
        dataset->num_wavelengths, dataset->width, dataset->height, recon
    );

    std::cout << "Recon results:" << std::endl;
    std::cout << "  h_bscan size: " << recon.h_bscan.size() << std::endl;
    std::cout << "  v_bscan size: " << recon.v_bscan.size() << std::endl;
    std::cout << "  ascan size: " << recon.ascan.size() << std::endl;

    // Find peak in A-scan
    float max_ascan = recon.ascan[0];
    int max_k = 0;
    for (size_t k = 0; k < recon.ascan.size(); ++k) {
        if (recon.ascan[k] > max_ascan) {
            max_ascan = recon.ascan[k];
            max_k = (int)k;
        }
    }
    std::cout << "  Central A-scan Peak: " << max_ascan << " dB at depth bin k=" << max_k << std::endl;

    return 0;
}

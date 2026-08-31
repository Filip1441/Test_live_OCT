#include "dataset_loader.hpp"
#include "config.hpp"
#include <zstd.h>
#include <dispatch/dispatch.h>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <numeric>

namespace oct {

std::shared_ptr<Dataset> DatasetLoader::load(const std::string& zarr_path, const std::string& uuid) {
    namespace fs = std::filesystem;
    fs::path base_path = zarr_path;
    fs::path level0_path = base_path / "0" / "0" / "0";

    if (!fs::exists(level0_path)) {
        std::cout << "[DatasetLoader] OME-ZARR path " << level0_path << " not found, generating synthetic phantom." << std::endl;
        return create_synthetic(SWEEP_POINTS, RAW_HEIGHT, RAW_WIDTH);
    }

    auto dataset = std::make_shared<Dataset>();
    dataset->num_wavelengths = SWEEP_POINTS;
    dataset->height = RAW_HEIGHT;
    dataset->width = RAW_WIDTH;
    dataset->uuid = uuid;

    // Allocate 6.79 GB buffer
    const size_t total_elements = (size_t)SWEEP_POINTS * RAW_HEIGHT * RAW_WIDTH;
    dataset->data.resize(total_elements, 0.0f);

    // Initialize wavelength vector (803.0 nm .. 878.0 nm)
    dataset->wavelengths_nm.resize(SWEEP_POINTS);
    for (int i = 0; i < SWEEP_POINTS; ++i) {
        dataset->wavelengths_nm[i] = 803.0 + i * (75.0 / (SWEEP_POINTS - 1));
    }
    dataset->lambda_step_nm = (float)((878.0 - 803.0) / (SWEEP_POINTS - 1));

    // Collect all chunk tasks: (zc: 0..23, yc: 0..5, xc: 0..5) -> 864 chunks
    struct ChunkTask {
        int zc, yc, xc;
        std::string filepath;
    };
    std::vector<ChunkTask> tasks;

    for (int zc = 0; zc < 24; ++zc) {
        for (int yc = 0; yc < 6; ++yc) {
            for (int xc = 0; xc < 6; ++xc) {
                fs::path chunk_file = level0_path / std::to_string(zc) / std::to_string(yc) / std::to_string(xc);
                if (fs::exists(chunk_file)) {
                    tasks.push_back({zc, yc, xc, chunk_file.string()});
                }
            }
        }
    }

    std::cout << "[DatasetLoader] Found " << tasks.size() << " chunks. Decompressing in parallel with GCD..." << std::endl;
    float* __restrict__ global_vol = dataset->data.data();

    // Parallel multi-core decompression using Apple Grand Central Dispatch
    dispatch_apply(tasks.size(), dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0), ^(size_t i) {
        const auto& task = tasks[i];
        std::ifstream file(task.filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return;

        std::streamsize compressed_size = file.tellg();
        file.seekg(0, std::ios::beg);

        std::vector<char> compressed_buf(compressed_size);
        if (!file.read(compressed_buf.data(), compressed_size)) return;

        // Target chunk is 32 x 256 x 256 floats (2 MB)
        constexpr size_t chunk_z_dim = 32;
        constexpr size_t chunk_y_dim = 256;
        constexpr size_t chunk_x_dim = 256;
        constexpr size_t chunk_elements = chunk_z_dim * chunk_y_dim * chunk_x_dim;
        std::vector<float> decompressed_buf(chunk_elements);

        size_t decompressed_size = ZSTD_decompress(
            decompressed_buf.data(),
            chunk_elements * sizeof(float),
            compressed_buf.data(),
            compressed_size
        );

        if (ZSTD_isError(decompressed_size)) return;

        // Copy chunk slices into main 3D volume
        const int z_start = task.zc * 32;
        const int y_start = task.yc * 256;
        const int x_start = task.xc * 256;

        const int z_count = std::min(32, SWEEP_POINTS - z_start);
        const int y_count = std::min(256, RAW_HEIGHT - y_start);
        const int x_count = std::min(256, RAW_WIDTH - x_start);

        for (int dz = 0; dz < z_count; ++dz) {
            int z = z_start + dz;
            for (int dy = 0; dy < y_count; ++dy) {
                int y = y_start + dy;
                float* dst = global_vol + ((size_t)z * RAW_HEIGHT + y) * RAW_WIDTH + x_start;
                const float* src = decompressed_buf.data() + (dz * chunk_y_dim + dy) * chunk_x_dim;
                std::memcpy(dst, src, x_count * sizeof(float));
            }
        }
    });

    std::cout << "[DatasetLoader] Successfully loaded full uncropped dataset (" 
              << SWEEP_POINTS << " x " << RAW_HEIGHT << " x " << RAW_WIDTH << ")." << std::endl;
    return dataset;
}

std::shared_ptr<Dataset> DatasetLoader::create_synthetic(int num_wavelengths, int height, int width) {
    auto dataset = std::make_shared<Dataset>();
    dataset->num_wavelengths = num_wavelengths;
    dataset->height = height;
    dataset->width = width;
    dataset->uuid = "synthetic-oct-phantom";

    const size_t total_elements = (size_t)num_wavelengths * height * width;
    dataset->data.resize(total_elements, 0.0f);

    dataset->wavelengths_nm.resize(num_wavelengths);
    for (int i = 0; i < num_wavelengths; ++i) {
        dataset->wavelengths_nm[i] = 803.0 + i * (75.0 / (num_wavelengths - 1));
    }
    dataset->lambda_step_nm = (float)((878.0 - 803.0) / (num_wavelengths - 1));

    float* vol = dataset->data.data();

    // Generate 3 reflective depth layers in volume
    const std::vector<float> layer_depths = {0.15f, 0.35f, 0.65f};
    const std::vector<float> layer_amps = {1.0f, 0.7f, 0.4f};

    dispatch_apply(num_wavelengths, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0), ^(size_t l) {
        float kn = (float)l / (float)(num_wavelengths - 1);
        float* frame = vol + l * ((size_t)height * width);

        for (int y = 0; y < height; ++y) {
            float ny = (float)y / (float)height - 0.5f;
            for (int x = 0; x < width; ++x) {
                float nx = (float)x / (float)width - 0.5f;
                float r = std::sqrt(nx * nx + ny * ny);

                float val = 10.0f; // DC background
                for (size_t k = 0; k < layer_depths.size(); ++k) {
                    float z = layer_depths[k];
                    float amp = layer_amps[k];
                    float mod = std::exp(-((r - z * 0.4f) * (r - z * 0.4f)) / 0.05f) * amp;
                    float phase = 2.0f * 3.14159265f * kn * (z * 100.0f);
                    val += std::cos(phase) * mod;
                }
                frame[y * width + x] = val;
            }
        }
    });

    return dataset;
}

} // namespace oct

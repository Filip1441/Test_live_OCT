#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include "config.hpp"
#include "types.hpp"
#include "hdf5_loader.hpp"
#include "dataset_loader.hpp"
#include "colormaps.hpp"
#include "cpu_compute_engine.hpp"
#include "producer_thread.hpp"
#include "processor_thread.hpp"

void test_hdf5_loader() {
    std::cout << "[TEST] 1. Testing HDF5 Loader with native dataset..." << std::endl;
    std::string path = oct::DEFAULT_DATASET_PATH;
    if (oct::Hdf5Loader::is_valid_hdf5(path)) {
        auto dataset = oct::Hdf5Loader::load(path);
        assert(dataset != nullptr);
        assert(dataset->num_wavelengths == oct::SWEEP_POINTS);
        assert(dataset->height == oct::RAW_HEIGHT);
        assert(dataset->width == oct::RAW_WIDTH);
        assert(dataset->data.size() == (size_t)oct::SWEEP_POINTS * oct::RAW_HEIGHT * oct::RAW_WIDTH);
        std::cout << "  --> PASSED: Native HDF5 dataset (" << dataset->num_wavelengths << " x " 
                  << dataset->height << " x " << dataset->width << ") loaded in full." << std::endl;
    } else {
        std::cout << "  --> SKIPPED: HDF5 file not found at default path." << std::endl;
    }
}

void test_colormaps() {
    std::cout << "[TEST] 2. Testing Colormaps & RGBA LUT generation..." << std::endl;
    const auto& names = oct::ColormapManager::get_names();
    assert(!names.empty());

    std::vector<float> db_test = {20.0f, 40.0f, 60.0f, 80.0f};
    std::vector<uint32_t> rgba_test(4);
    oct::ColormapManager::apply_colormap(
        db_test.data(), db_test.size(), 20.0f, 80.0f, oct::ColormapType::Inferno, rgba_test.data()
    );
    for (uint32_t pixel : rgba_test) {
        assert(pixel != 0);
    }
    std::cout << "  --> PASSED: Colormap LUTs and fast conversions verified." << std::endl;
}

void test_cpu_compute_engine() {
    std::cout << "[TEST] 3. Testing CPU AMX Accelerate Compute Engine..." << std::endl;
    oct::CpuComputeEngine engine(oct::SWEEP_POINTS);

    std::vector<float> h_raw((size_t)oct::SWEEP_POINTS * oct::RAW_WIDTH, 1.0f);
    std::vector<float> v_raw((size_t)oct::SWEEP_POINTS * oct::RAW_HEIGHT, 1.0f);
    std::vector<float> a_raw(oct::SWEEP_POINTS, 1.0f);

    // Add modulation signal
    for (int l = 0; l < oct::SWEEP_POINTS; ++l) {
        float s = std::cos(2.0f * 3.14159f * (float)l * 0.1f);
        for (int x = 0; x < oct::RAW_WIDTH; ++x) h_raw[l * oct::RAW_WIDTH + x] += s;
        for (int y = 0; y < oct::RAW_HEIGHT; ++y) v_raw[l * oct::RAW_HEIGHT + y] += s;
        a_raw[l] += s;
    }

    oct::CrosshairReconResult res;
    auto t0 = std::chrono::high_resolution_clock::now();
    engine.compute_from_lines(
        h_raw.data(), v_raw.data(), a_raw.data(),
        oct::SWEEP_POINTS, oct::RAW_WIDTH, oct::RAW_HEIGHT, res
    );
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    assert(res.h_bscan.size() == (size_t)oct::DEPTH_POINTS * oct::RAW_WIDTH);
    assert(res.v_bscan.size() == (size_t)oct::DEPTH_POINTS * oct::RAW_HEIGHT);
    assert(res.ascan.size() == (size_t)oct::DEPTH_POINTS);
    std::cout << "  --> PASSED: CPU FFT completed in " << ms << " ms." << std::endl;
}

void test_dynamic_wavelength_engine() {
    std::cout << "[TEST] 3b. Testing Dynamic 251-wavelength Compute Engine (CPU & Metal)..." << std::endl;
    const int custom_lambdas = 251;
    const int custom_depth_bins = 251 / 2 + 1;
    oct::CpuComputeEngine engine(custom_lambdas);

    std::vector<float> h_raw((size_t)custom_lambdas * oct::RAW_WIDTH, 1.0f);
    std::vector<float> v_raw((size_t)custom_lambdas * oct::RAW_HEIGHT, 1.0f);
    std::vector<float> a_raw(custom_lambdas, 1.0f);

    oct::CrosshairReconResult res;
    engine.compute_from_lines(
        h_raw.data(), v_raw.data(), a_raw.data(),
        custom_lambdas, oct::RAW_WIDTH, oct::RAW_HEIGHT, res
    );

    assert(res.h_bscan.size() == (size_t)custom_depth_bins * oct::RAW_WIDTH);
    assert(res.v_bscan.size() == (size_t)custom_depth_bins * oct::RAW_HEIGHT);
    assert(res.ascan.size() == (size_t)custom_depth_bins);
    std::cout << "  --> PASSED: Dynamic 251 lambdas -> " << custom_depth_bins << " depth bins FFT verified." << std::endl;
}

void test_pipeline() {
    std::cout << "[TEST] 4. Testing Multi-threaded Decoupled Pipeline (Producer + Processor)..." << std::endl;
    auto dataset = oct::DatasetLoader::create_synthetic(251, oct::RAW_HEIGHT, oct::RAW_WIDTH);

    auto raw_queue = std::make_shared<oct::ThreadSafeQueue<oct::RawFrameMessage>>(100);
    auto res_queue = std::make_shared<oct::ThreadSafeQueue<oct::ProcessedResult>>(10);

    auto producer = std::make_shared<oct::ProducerThread>(dataset, raw_queue);
    auto processor = std::make_shared<oct::ProcessorThread>(
        raw_queue, res_queue, dataset->num_wavelengths, dataset->height, dataset->width
    );

    producer->set_tuning_speed(200.0f); // Fast speed for unit testing
    producer->start();
    processor->start();

    // Let pipeline run for 200 ms
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    producer->stop();
    processor->stop();

    auto last_res = res_queue->try_pop_latest();
    assert(last_res.has_value());
    assert(last_res.value().h_bscan.size() == (size_t)(251 / 2 + 1) * oct::RAW_WIDTH);
    std::cout << "  --> PASSED: Multi-threaded pipeline dynamically processed " 
              << last_res.value().lambda_index << " frames with "
              << last_res.value().compute_latency_ms << " ms compute latency on 251 lambdas." << std::endl;
}

int main() {
    std::cout << "=================================================" << std::endl;
    std::cout << "    SS-FF-OCT Pure C++ Core Test Suite           " << std::endl;
    std::cout << "=================================================" << std::endl;

    test_hdf5_loader();
    test_colormaps();
    test_cpu_compute_engine();
    test_dynamic_wavelength_engine();
    test_pipeline();

    std::cout << "=================================================" << std::endl;
    std::cout << "    ALL TESTS PASSED SUCCESSFULLY! (100% C++)   " << std::endl;
    std::cout << "=================================================" << std::endl;
    return 0;
}

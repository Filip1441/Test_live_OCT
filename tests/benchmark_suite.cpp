#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include "config.hpp"
#include "types.hpp"
#include "dataset_loader.hpp"
#include "cpu_compute_engine.hpp"
#include "metal_compute_engine.hpp"
#include "producer_thread.hpp"
#include "processor_thread.hpp"

struct KernelBenchmarkResult {
    std::string backend_name;
    double avg_latency_ms;
    double min_latency_ms;
    double max_latency_ms;
    double p95_latency_ms;
    double max_theoretical_fps;
};

struct PipelineBenchmarkResult {
    double tuning_speed_nm_s;
    double target_fps;
    std::string approach_name;
    std::string backend_name;
    double producer_fps;
    double processor_fps;
    double avg_compute_latency_ms;
    uint64_t frames_emitted;
    uint64_t frames_processed;
    uint64_t frames_dropped;
    bool is_realtime_stable;
};

KernelBenchmarkResult benchmark_kernel(oct::IComputeEngine* engine, const std::string& name, int iterations = 100) {
    std::vector<float> h_raw((size_t)oct::SWEEP_POINTS * oct::RAW_WIDTH, 1.0f);
    std::vector<float> v_raw((size_t)oct::SWEEP_POINTS * oct::RAW_HEIGHT, 1.0f);
    std::vector<float> a_raw(oct::SWEEP_POINTS, 1.0f);

    // Warmup
    oct::CrosshairReconResult dummy_res;
    for (int i = 0; i < 5; ++i) {
        engine->compute_from_lines(
            h_raw.data(), v_raw.data(), a_raw.data(),
            oct::SWEEP_POINTS, oct::RAW_WIDTH, oct::RAW_HEIGHT, dummy_res
        );
    }

    std::vector<double> latencies;
    latencies.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto t0 = std::chrono::high_resolution_clock::now();
        engine->compute_from_lines(
            h_raw.data(), v_raw.data(), a_raw.data(),
            oct::SWEEP_POINTS, oct::RAW_WIDTH, oct::RAW_HEIGHT, dummy_res
        );
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        latencies.push_back(ms);
    }

    std::sort(latencies.begin(), latencies.end());
    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg = sum / iterations;
    double min_v = latencies.front();
    double max_v = latencies.back();
    double p95 = latencies[(size_t)(iterations * 0.95)];
    double theoretical_fps = 1000.0 / avg;

    return {name, avg, min_v, max_v, p95, theoretical_fps};
}

PipelineBenchmarkResult benchmark_pipeline(
    std::shared_ptr<oct::Dataset> dataset,
    double speed_nm_s,
    oct::ProcessingApproach approach,
    oct::ComputeBackend backend,
    double duration_sec = 2.0
) {
    auto raw_queue = std::make_shared<oct::ThreadSafeQueue<oct::RawFrameMessage>>(oct::RAW_FRAME_QUEUE_MAXSIZE);
    auto res_queue = std::make_shared<oct::ThreadSafeQueue<oct::ProcessedResult>>(oct::RESULTS_QUEUE_MAXSIZE);

    oct::ProducerThread producer(dataset, raw_queue);
    oct::ProcessorThread processor(raw_queue, res_queue, dataset->num_wavelengths, dataset->height, dataset->width);

    producer.set_tuning_speed((float)speed_nm_s);
    producer.set_simulate_drift(false);
    producer.set_simulate_drops(false);

    processor.set_approach(approach);
    processor.set_backend(backend);

    producer.start();
    processor.start();

    // Run for duration
    std::this_thread::sleep_for(std::chrono::duration<double>(duration_sec));

    producer.stop();
    processor.stop();

    auto prod_stats = producer.get_stats();
    auto proc_stats = processor.get_stats();

    double target_fps = speed_nm_s / 0.1;
    std::string approach_str = (approach == oct::ProcessingApproach::RollingBuffer) ? "Rolling Buffer" : "Full Sweep";
    std::string backend_str = (backend == oct::ComputeBackend::MetalGPU) ? "Metal GPU" : "CPU AMX";

    bool stable = (proc_stats.fps >= target_fps * 0.85 || proc_stats.fps >= 100.0);

    return {
        speed_nm_s,
        target_fps,
        approach_str,
        backend_str,
        prod_stats.fps,
        proc_stats.fps,
        proc_stats.last_compute_latency_ms,
        prod_stats.frames_emitted,
        proc_stats.frames_processed,
        prod_stats.frames_dropped,
        stable
    };
}

int main() {
    std::cout << "==========================================================================" << std::endl;
    std::cout << "        SS-FF-OCT Comprehensive HPC Performance Benchmark Suite           " << std::endl;
    std::cout << "==========================================================================" << std::endl;
    std::cout << "System: Apple Silicon Native (arm64, C++20, Metal Shaders, AMX Accelerate)" << std::endl;
    std::cout << "Dataset Dimensions: 751 lambdas x 1504 H x 1504 W (6.79 GB uncropped)" << std::endl;
    std::cout << "==========================================================================\n" << std::endl;

    // 1. Benchmark Isolated Compute Engines
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << " Phase 1: Pure Compute Engine Latency Benchmark (100 Iterations)" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    oct::CpuComputeEngine cpu_engine(oct::SWEEP_POINTS);
    oct::MetalComputeEngine metal_engine(oct::SWEEP_POINTS);

    auto cpu_res = benchmark_kernel(&cpu_engine, "CPU AMX Accelerate (vDSP/SGEMM)", 100);
    auto gpu_res = benchmark_kernel(&metal_engine, "Apple Metal GPU Compute Pipeline", 100);

    std::cout << std::fixed << std::setprecision(2);
    std::cout << "| Backend Engine                  | Avg Latency | Min Latency | Max Latency | p95 Latency | Max Compute Throughput |" << std::endl;
    std::cout << "|---------------------------------|-------------|-------------|-------------|-------------|------------------------|" << std::endl;
    std::cout << "| " << std::left << std::setw(31) << cpu_res.backend_name
              << " | " << std::setw(9) << cpu_res.avg_latency_ms << " ms"
              << " | " << std::setw(9) << cpu_res.min_latency_ms << " ms"
              << " | " << std::setw(9) << cpu_res.max_latency_ms << " ms"
              << " | " << std::setw(9) << cpu_res.p95_latency_ms << " ms"
              << " | " << std::setw(17) << cpu_res.max_theoretical_fps << " FPS |" << std::endl;

    std::cout << "| " << std::left << std::setw(31) << gpu_res.backend_name
              << " | " << std::setw(9) << gpu_res.avg_latency_ms << " ms"
              << " | " << std::setw(9) << gpu_res.min_latency_ms << " ms"
              << " | " << std::setw(9) << gpu_res.max_latency_ms << " ms"
              << " | " << std::setw(9) << gpu_res.p95_latency_ms << " ms"
              << " | " << std::setw(17) << gpu_res.max_theoretical_fps << " FPS |" << std::endl;
    std::cout << "--------------------------------------------------------------------------\n" << std::endl;

    // 2. Multi-threaded End-to-End Pipeline Benchmarks across Tuning Speeds
    std::cout << "--------------------------------------------------------------------------" << std::endl;
    std::cout << " Phase 2: Live Multi-Threaded Pipeline Under Load Across Tuning Speeds" << std::endl;
    std::cout << "--------------------------------------------------------------------------" << std::endl;

    std::cout << "[Loading OME-ZARR Dataset into resident memory...]" << std::endl;
    auto dataset = oct::DatasetLoader::load(oct::DEFAULT_DATASET_PATH, oct::DEFAULT_DATASET_UUID);

    std::vector<double> test_speeds = {5.0, 20.0, 75.0, 200.0, 500.0};
    std::vector<PipelineBenchmarkResult> results;

    for (double speed : test_speeds) {
        // Test CPU Rolling
        results.push_back(benchmark_pipeline(dataset, speed, oct::ProcessingApproach::RollingBuffer, oct::ComputeBackend::CPU, 1.5));
        // Test GPU Rolling
        results.push_back(benchmark_pipeline(dataset, speed, oct::ProcessingApproach::RollingBuffer, oct::ComputeBackend::MetalGPU, 1.5));
        // Test CPU Full Sweep
        results.push_back(benchmark_pipeline(dataset, speed, oct::ProcessingApproach::FullSweep, oct::ComputeBackend::CPU, 1.5));
        // Test GPU Full Sweep
        results.push_back(benchmark_pipeline(dataset, speed, oct::ProcessingApproach::FullSweep, oct::ComputeBackend::MetalGPU, 1.5));
    }

    std::cout << "\n==========================================================================================================" << std::endl;
    std::cout << "| Speed (nm/s) | Target FPS | Approach       | Backend   | Producer FPS | Processor FPS | Compute Latency | Status  |" << std::endl;
    std::cout << "|--------------|------------|----------------|-----------|--------------|---------------|-----------------|---------|" << std::endl;

    for (const auto& r : results) {
        std::cout << "| " << std::setw(10) << std::right << r.tuning_speed_nm_s << " nm/s"
                  << " | " << std::setw(8) << (int)r.target_fps << " FPS"
                  << " | " << std::left << std::setw(14) << r.approach_name
                  << " | " << std::left << std::setw(9) << r.backend_name
                  << " | " << std::setw(10) << std::right << r.producer_fps << " FPS"
                  << " | " << std::setw(11) << std::right << r.processor_fps << " FPS"
                  << " | " << std::setw(13) << std::right << r.avg_compute_latency_ms << " ms"
                  << " | " << (r.is_realtime_stable ? "STABLE  " : "CAPPED  ") << " |" << std::endl;
    }
    std::cout << "==========================================================================================================" << std::endl;

    return 0;
}

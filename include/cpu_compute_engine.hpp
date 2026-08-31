#pragma once

#include "compute_engine.hpp"
#include <vector>

namespace oct {

class SgemmFftPlan;

class CpuComputeEngine : public IComputeEngine {
public:
    explicit CpuComputeEngine(int num_lambdas = SWEEP_POINTS);
    ~CpuComputeEngine() override;

    ComputeBackend get_backend_type() const override { return ComputeBackend::CPU; }

    void compute_from_lines(
        const float* __restrict__ h_raw,
        const float* __restrict__ v_raw,
        const float* __restrict__ ascan_raw,
        int n_lambda,
        int width,
        int height,
        CrosshairReconResult& out_result
    ) override;

private:
    int num_lambdas_;
    int depth_bins_;
    std::vector<float> window_;
    std::unique_ptr<SgemmFftPlan> plan_;
};

} // namespace oct

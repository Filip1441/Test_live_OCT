#pragma once

#include <memory>
#include <vector>
#include <string>
#import <Metal/Metal.h>
#include "imgui.h"
#include "implot.h"
#include "types.hpp"
#include "config.hpp"
#include "colormaps.hpp"
#include "producer_thread.hpp"
#include "processor_thread.hpp"
#include "volume3d_engine.hpp"
#include "thread_safe_queue.hpp"

namespace oct {

class AppGUI {
public:
    AppGUI(
        id<MTLDevice> device,
        std::shared_ptr<ProducerThread> producer,
        std::shared_ptr<ProcessorThread> processor,
        std::shared_ptr<ThreadSafeQueue<ProcessedResult>> results_queue,
        PipelineMode pipeline_mode = PipelineMode::Optimized_DTFT
    );
    ~AppGUI();

    void update_and_render();

private:
    void render_top_bar();
    void render_ortho_viewer();
    void render_volume3d_viewer();
    void render_control_sidebar();

    void update_textures(const ProcessedResult& result);
    void recreate_textures(int num_lambdas, int height, int width);
    id<MTLTexture> create_texture(int width, int height);

    id<MTLDevice> device_;
    std::shared_ptr<ProducerThread> producer_;
    std::shared_ptr<ProcessorThread> processor_;
    std::shared_ptr<ThreadSafeQueue<ProcessedResult>> results_queue_;

    // Metal Textures for 2D & Ortho visualization
    id<MTLTexture> camera_texture_ = nil;
    id<MTLTexture> hbscan_texture_ = nil;
    id<MTLTexture> vbscan_texture_ = nil;

    id<MTLTexture> phase_texture_ = nil;
    id<MTLTexture> phase_colorbar_texture_ = nil;

    std::vector<uint32_t> camera_rgba_;
    std::vector<uint32_t> hbscan_rgba_;
    std::vector<uint32_t> vbscan_rgba_;
    std::vector<uint32_t> phase_rgba_;

    // 3D Volume & Interactive Raymarching Engine
    std::unique_ptr<Volume3DEngine> volume3d_engine_;

    // 3D Volume Camera & Raymarching State
    float raymarch_yaw_ = 0.75f;
    float raymarch_pitch_ = 1.25f;
    float raymarch_zoom_ = 1.8f;
    float raymarch_z_stretch_ = 3.5f;

    ComputeBackend3D volume3d_backend_ = ComputeBackend3D::CPUMultiCore;
    int volume3d_backend_idx_ = 1;
    bool live_continuous_recon_ = true;
    Volume3DStats last_3d_stats_;
    PipelineMode pipeline_mode_;

    // Phase-OCT Parameters & Range
    float phase_dz_focus_um_ = 0.0f;
    float phase_n_immersion_ = 1.52f;
    float phase_dx_um_ = 0.5f;
    float phase_cmin_ = -4.0f;
    float phase_cmax_ = 3.5f;

    // Interactive 1D Line Cut (normalized [0, 1])
    ImVec2 line_p0_ = ImVec2(0.1f, 0.5f);
    ImVec2 line_p1_ = ImVec2(0.9f, 0.5f);
    bool is_dragging_line_ = false;

    // Visualization settings
    float db_min_ = DEFAULT_DB_MIN;
    float db_max_ = DEFAULT_DB_MAX;
    ColormapType current_colormap_ = ColormapType::Inferno;
    int current_colormap_idx_ = 0;

    // Active state
    ProcessedResult latest_result_;
    bool has_result_ = false;

    std::string sample_file_path_;
    std::string reference_file_path_;

    void trigger_phase_reconstruction();
    void update_phase_textures();
};

} // namespace oct

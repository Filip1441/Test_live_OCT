#include "app_gui.hpp"
#include "hdf5_loader.hpp"
#include "file_dialog.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <filesystem>

namespace oct {

AppGUI::AppGUI(
    id<MTLDevice> device,
    std::shared_ptr<ProducerThread> producer,
    std::shared_ptr<ProcessorThread> processor,
    std::shared_ptr<ThreadSafeQueue<ProcessedResult>> results_queue,
    PipelineMode pipeline_mode
) : device_(device),
    producer_(producer),
    processor_(processor),
    results_queue_(results_queue),
    pipeline_mode_(pipeline_mode) {

    int num_lambdas = processor_->get_num_lambdas();
    int height = processor_->get_height();
    int width = processor_->get_width();
    int depth_bins = num_lambdas / 2 + 1;

    camera_rgba_.resize((size_t)height * width);
    hbscan_rgba_.resize((size_t)width * depth_bins);
    vbscan_rgba_.resize((size_t)depth_bins * height);

    camera_texture_ = create_texture(width, height);
    hbscan_texture_ = create_texture(width, depth_bins);
    vbscan_texture_ = create_texture(depth_bins, height);

    volume3d_engine_ = std::make_unique<Volume3DEngine>(device_, width, height, num_lambdas);
    volume3d_backend_ = ComputeBackend3D::CPUMultiCore;
    volume3d_backend_idx_ = 1;
}

AppGUI::~AppGUI() {
    camera_texture_ = nil;
    hbscan_texture_ = nil;
    vbscan_texture_ = nil;
    phase_texture_ = nil;
    phase_colorbar_texture_ = nil;
}

void AppGUI::recreate_textures(int num_lambdas, int height, int width) {
    int depth_bins = num_lambdas / 2 + 1;
    camera_rgba_.resize((size_t)height * width);
    hbscan_rgba_.resize((size_t)width * depth_bins);
    vbscan_rgba_.resize((size_t)depth_bins * height);

    camera_texture_ = create_texture(width, height);
    hbscan_texture_ = create_texture(width, depth_bins);
    vbscan_texture_ = create_texture(depth_bins, height);

    phase_texture_ = create_texture(width, height);
}

id<MTLTexture> AppGUI::create_texture(int width, int height) {
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                   width:width
                                                                                  height:height
                                                                               mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    return [device_ newTextureWithDescriptor:desc];
}

void AppGUI::update_textures(const ProcessedResult& result) {
    int w = processor_->get_width();
    int h = processor_->get_height();
    int num_lambdas = processor_->get_num_lambdas();
    int depth_bins = num_lambdas / 2 + 1;

    // 1. Update 2D Camera Image (Grayscale mapped with robust auto-contrast)
    if (!result.camera_frame.empty()) {
        float min_v = result.camera_frame[0];
        float max_v = result.camera_frame[0];
        for (float v : result.camera_frame) {
            min_v = std::min(min_v, v);
            max_v = std::max(max_v, v);
        }
        ColormapManager::apply_colormap(
            result.camera_frame.data(),
            result.camera_frame.size(),
            min_v,
            max_v + 1e-3f,
            ColormapType::Grayscale,
            camera_rgba_.data()
        );
        [camera_texture_ replaceRegion:MTLRegionMake2D(0, 0, w, h)
                           mipmapLevel:0
                             withBytes:camera_rgba_.data()
                           bytesPerRow:w * 4];
    }

    // 2. Update Horizontal B-Scan [Depth_bins, Width]
    if (!result.h_bscan.empty()) {
        ColormapManager::apply_colormap(
            result.h_bscan.data(),
            result.h_bscan.size(),
            db_min_,
            db_max_,
            current_colormap_,
            hbscan_rgba_.data()
        );
        [hbscan_texture_ replaceRegion:MTLRegionMake2D(0, 0, w, depth_bins)
                           mipmapLevel:0
                             withBytes:hbscan_rgba_.data()
                           bytesPerRow:w * 4];
    }

    // 3. Update Vertical B-Scan [Height, Depth_bins]
    if (!result.v_bscan.empty()) {
        ColormapManager::apply_colormap(
            result.v_bscan.data(),
            result.v_bscan.size(),
            db_min_,
            db_max_,
            current_colormap_,
            vbscan_rgba_.data()
        );
        [vbscan_texture_ replaceRegion:MTLRegionMake2D(0, 0, depth_bins, h)
                           mipmapLevel:0
                             withBytes:vbscan_rgba_.data()
                           bytesPerRow:depth_bins * 4];
    }
}

void AppGUI::trigger_phase_reconstruction() {
    int obj_w = processor_->get_width();
    int obj_h = processor_->get_height();
    int obj_nl = processor_->get_num_lambdas();

    volume3d_engine_->reinitialize(obj_w, obj_h, obj_nl);

    const float* vol_obj = processor_->get_volume_data();
    const float* vol_ref = nullptr;
    if (processor_->has_reference_dataset()) {
        auto ref_ds = processor_->get_reference_dataset();
        if (ref_ds && !ref_ds->data.empty() &&
            ref_ds->width == obj_w && ref_ds->height == obj_h && ref_ds->num_wavelengths == obj_nl) {
            vol_ref = ref_ds->data.data();
        }
    }

    std::vector<double> wavelengths = processor_->get_wavelengths();
    int n_lambdas = obj_nl;
    if (wavelengths.size() != (size_t)n_lambdas) {
        wavelengths.resize(n_lambdas);
        for (int i = 0; i < n_lambdas; ++i) {
            float frac = (n_lambdas > 1) ? (float)i / (float)(n_lambdas - 1) : 0.0f;
            wavelengths[i] = 803.0 + frac * (878.0 - 803.0);
        }
    }

    if (vol_obj) {
        volume3d_engine_->reconstruct_phase_oct(
            vol_obj, vol_ref, wavelengths,
            phase_dz_focus_um_, phase_n_immersion_, phase_dx_um_,
            false, pipeline_mode_, volume3d_backend_, last_3d_stats_
        );
        update_phase_textures();
    }
}

void AppGUI::update_phase_textures() {
    if (!volume3d_engine_ || !volume3d_engine_->is_volume_reconstructed()) return;
    const auto& ph = volume3d_engine_->get_phase_reconstructed();
    int w = processor_->get_width();
    int h = processor_->get_height();
    if (ph.empty()) return;

    if (!phase_texture_ || [phase_texture_ width] != w || [phase_texture_ height] != h) {
        phase_texture_ = create_texture(w, h);
    }
    phase_rgba_.resize((size_t)w * h);

    const uint32_t* lut_jet = ColormapManager::get_lut(ColormapType::Jet);
    float range = (phase_cmax_ > phase_cmin_) ? (phase_cmax_ - phase_cmin_) : 1.0f;
    for (size_t i = 0; i < (size_t)w * h; ++i) {
        float norm = std::clamp((ph[i] - phase_cmin_) / range, 0.0f, 1.0f);
        int idx = std::clamp((int)(norm * 255.0f), 0, 255);
        phase_rgba_[i] = lut_jet[idx];
    }
    [phase_texture_ replaceRegion:MTLRegionMake2D(0, 0, w, h)
                      mipmapLevel:0
                        withBytes:phase_rgba_.data()
                      bytesPerRow:w * sizeof(uint32_t)];

    if (!phase_colorbar_texture_) {
        phase_colorbar_texture_ = create_texture(1, 256);
        std::vector<uint32_t> cb(256);
        for (int y = 0; y < 256; ++y) cb[y] = lut_jet[255 - y];
        [phase_colorbar_texture_ replaceRegion:MTLRegionMake2D(0, 0, 1, 256)
                                   mipmapLevel:0
                                     withBytes:cb.data()
                                   bytesPerRow:sizeof(uint32_t)];
    }
}

void AppGUI::update_and_render() {
    // Non-blocking dequeue of latest streaming frame
    auto opt_res = results_queue_->try_pop_latest();
    if (opt_res.has_value()) {
        latest_result_ = std::move(opt_res.value());
        has_result_ = true;
        update_textures(latest_result_);

        // Live Continuous Phase & 3D Topography Update (so moving sample updates the final result!)
        if (live_continuous_recon_) {
            static int recon_throttle = 0;
            bool should_live_update = false;
            if (processor_->get_approach() == ProcessingApproach::RollingBuffer) {
                if (++recon_throttle >= 6) {
                    recon_throttle = 0;
                    should_live_update = true;
                }
            } else {
                if (latest_result_.lambda_index == 0) {
                    should_live_update = true;
                }
            }
            if (should_live_update) {
                trigger_phase_reconstruction();
            }
        }
    }

    // Fullscreen Root Window
    ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(viewport->Pos);
    ImGui::SetNextWindowSize(viewport->Size);

    ImGuiWindowFlags window_flags = ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoCollapse |
                                    ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove |
                                    ImGuiWindowFlags_NoBringToFrontOnFocus | ImGuiWindowFlags_NoNavFocus;

    ImGui::PushStyleVar(ImGuiStyleVar_WindowRounding, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowBorderSize, 0.0f);
    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 8.0f));

    ImGui::Begin("SS-FF-OCT Real-time Simulator", nullptr, window_flags);
    ImGui::PopStyleVar(3);

    // =============================================================
    // 1. TOP HEADER BANNER (Telemetry, Laser Controls, Sim Toggles)
    // =============================================================
    ImGui::BeginChild("HeaderBanner", ImVec2(0, 130), true);
    ImGui::Columns(4, "HeaderCols", false);

    // Col 0: Streaming Telemetry
    auto p_stats = producer_->get_stats();
    auto proc_stats = processor_->get_stats();
    ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Live Streaming Telemetry:");
    ImGui::Text("Laser: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%.1f nm (Sweep #%d)", p_stats.current_wavelength_nm, p_stats.current_sweep);
    ImGui::Text("Producer: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "%.0f FPS", p_stats.fps);
    ImGui::SameLine();
    ImGui::Text("| Proc: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "%.0f FPS", proc_stats.fps);

    ImGui::NextColumn();

    // Col 1: Phase-OCT 3D Reconstruction Telemetry
    ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Phase-OCT Engine (3D):");
    ImGui::Text("Time (RAM->Phase): ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%.2f ms (%.3f s)", last_3d_stats_.compute_time_ms, last_3d_stats_.compute_time_ms / 1000.0f);
    ImGui::Text("z_obj: %d | z_ref: %d", last_3d_stats_.z_obj_index, last_3d_stats_.z_ref_index);
    ImGui::Text("Step Phase: ");
    ImGui::SameLine();
    ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Delta phi = %+.2f rad", last_3d_stats_.step_phase_delay_rad);
    ImGui::Checkbox("Live 3D Continuous Recon", &live_continuous_recon_);

    ImGui::NextColumn();

    // Col 2: Processing Approach & Laser Speed
    ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Processing & Laser Speed:");
    const char* approach_items[] = { "Approach 2: Rolling Buffer (Real-time)", "Approach 1: Full Sweep" };
    int current_approach_idx = (processor_->get_approach() == ProcessingApproach::RollingBuffer) ? 0 : 1;
    ImGui::SetNextItemWidth(210);
    if (ImGui::Combo("##Approach", &current_approach_idx, approach_items, IM_ARRAYSIZE(approach_items))) {
        processor_->set_approach((current_approach_idx == 0) ? ProcessingApproach::RollingBuffer : ProcessingApproach::FullSweep);
    }

    float speed = producer_->get_tuning_speed();
    ImGui::SetNextItemWidth(140);
    if (ImGui::SliderFloat("Speed", &speed, MIN_TUNING_SPEED_NM_S, MAX_TUNING_SPEED_NM_S, "%.1f nm/s")) {
        producer_->set_tuning_speed(speed);
    }

    ImGui::NextColumn();

    // Col 3: Playback Controls, File Loaders & Sim Toggles
    if (ImGui::Button("▶ Play", ImVec2(50, 22))) {
        producer_->resume();
        processor_->resume();
    }
    ImGui::SameLine();
    if (ImGui::Button("⏸ Pause", ImVec2(55, 22))) {
        producer_->pause();
        processor_->pause();
    }
    ImGui::SameLine();
    if (ImGui::Button("⚡ Recon 3D", ImVec2(85, 22))) {
        trigger_phase_reconstruction();
    }

    if (ImGui::Button("📁 Load Sample...", ImVec2(105, 22))) {
        std::string path = FileDialog::open_file("Select OCT Measurement File (*.hdf5)");
        if (!path.empty()) {
            try {
                auto new_ds = Hdf5Loader::load(path);
                producer_->set_dataset(new_ds);
                processor_->reinitialize(new_ds->num_wavelengths, new_ds->height, new_ds->width);
                volume3d_engine_->reinitialize(new_ds->width, new_ds->height, new_ds->num_wavelengths);
                recreate_textures(new_ds->num_wavelengths, new_ds->height, new_ds->width);
                sample_file_path_ = path;
                trigger_phase_reconstruction();
            } catch (const std::exception& ex) {
                std::cerr << "[GUI Error] Failed to load HDF5 sample: " << ex.what() << std::endl;
            }
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("📁 Load Ref...", ImVec2(90, 22))) {
        std::string path = FileDialog::open_file("Select OCT Reference File (*.hdf5)");
        if (!path.empty()) {
            try {
                auto ref_ds = Hdf5Loader::load(path, true);
                processor_->set_reference_dataset(ref_ds);
                reference_file_path_ = path;
                trigger_phase_reconstruction();
            } catch (const std::exception& ex) {
                std::cerr << "[GUI Error] Failed to load HDF5 reference: " << ex.what() << std::endl;
            }
        }
    }

    bool sim_drift = producer_->get_simulate_drift();
    if (ImGui::Checkbox("Drift Motion", &sim_drift)) {
        producer_->set_simulate_drift(sim_drift);
    }
    ImGui::SameLine();
    bool sim_drops = producer_->get_simulate_drops();
    if (ImGui::Checkbox("Drop Frames", &sim_drops)) {
        producer_->set_simulate_drops(sim_drops);
    }
    if (sim_drops) {
        ImGui::SameLine();
        float drop_pct = producer_->get_drop_rate() * 100.0f;
        ImGui::SetNextItemWidth(80);
        if (ImGui::SliderFloat("##DropRate", &drop_pct, 1.0f, 75.0f, "%.0f%%")) {
            producer_->set_drop_rate(drop_pct / 100.0f);
        }
    }

    ImGui::Columns(1);
    ImGui::EndChild();

    // =============================================================
    // 2. MAIN SPLIT VIEW (Left: 2D Stream & B-Scans | Right: Phase-OCT 2D & 3D Surface)
    // =============================================================
    float avail_w = ImGui::GetContentRegionAvail().x;
    float avail_h = ImGui::GetContentRegionAvail().y;
    float col_w = (avail_w - 12.0f) * 0.5f;

    // -------------------------------------------------------------
    // LEFT COLUMN: 2D Real-time Sweeping Stream & Orthogonal B-Scans
    // -------------------------------------------------------------
    ImGui::BeginChild("LeftStreamingPanel", ImVec2(col_w, avail_h), true);
    render_ortho_viewer();
    ImGui::EndChild();

    ImGui::SameLine();

    // -------------------------------------------------------------
    // RIGHT COLUMN: Phase-OCT 2D Unwrapped Map, Line Profile & 3D Topography Mesh
    // -------------------------------------------------------------
    ImGui::BeginChild("RightPhasePanel", ImVec2(col_w, avail_h), true);
    render_volume3d_viewer();
    ImGui::EndChild();

    ImGui::End();
}

void AppGUI::render_ortho_viewer() {
    float avail_w = ImGui::GetContentRegionAvail().x;
    float avail_h = ImGui::GetContentRegionAvail().y;

    int w = processor_->get_width();
    int h = processor_->get_height();

    float depth_thickness = 130.0f;
    float square_side = std::max(180.0f, std::min(avail_w - depth_thickness - 28.0f, (avail_h - depth_thickness - 60.0f) * 0.85f));

    CrosshairPos crosshair = processor_->get_crosshair();

    ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Live 2D Sweeping Stream & Orthogonal B-Scans (Without reference subtraction)");
    ImGui::Separator();

    // 1. 2D Camera En-Face View (Stationary, 1:1 Aspect Ratio, Draggable Crosshair)
    ImGui::BeginChild("CameraViewBox", ImVec2(square_side + 12.0f, square_side + 28.0f), true);
    ImGui::Text("2D Camera (%dx%d) | Crosshair: X=%d, Y=%d", w, h, crosshair.x, crosshair.y);

    ImVec2 img_pos = ImGui::GetCursorScreenPos();
    ImVec2 img_size(square_side, square_side);

    if (camera_texture_) {
        ImGui::Image((__bridge void*)camera_texture_, img_size);

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 mouse_pos = ImGui::GetMousePos();

        float ch_norm_x = (float)crosshair.x / (float)w;
        float ch_norm_y = (float)crosshair.y / (float)h;

        float screen_cx = img_pos.x + ch_norm_x * square_side;
        float screen_cy = img_pos.y + ch_norm_y * square_side;

        // Draw Crosshair Lines (Cyan Vertical, Yellow Horizontal)
        draw_list->AddLine(ImVec2(screen_cx, img_pos.y), ImVec2(screen_cx, img_pos.y + square_side),
                           IM_COL32(0, 240, 255, 230), 2.0f);
        draw_list->AddLine(ImVec2(img_pos.x, screen_cy), ImVec2(img_pos.x + square_side, screen_cy),
                           IM_COL32(255, 234, 0, 230), 2.0f);

        // Handle Direct Mouse Interaction to drag crosshairs
        if (ImGui::IsItemHovered()) {
            if (ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
                int new_x = (int)(((mouse_pos.x - img_pos.x) / square_side) * w);
                int new_y = (int)(((mouse_pos.y - img_pos.y) / square_side) * h);
                processor_->set_crosshair(new_x, new_y);
            }
        }
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // 2. Vertical B-Scan (Rotated along Y, exact same height as 2D camera image)
    ImGui::BeginChild("VBScanBox", ImVec2(depth_thickness + 12.0f, square_side + 28.0f), true);
    ImGui::Text("V-BScan (X=%d)", crosshair.x);
    ImVec2 vb_pos = ImGui::GetCursorScreenPos();
    ImVec2 vb_size(depth_thickness, square_side);

    if (vbscan_texture_) {
        ImGui::Image((__bridge void*)vbscan_texture_, vb_size);

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float norm_y = (float)crosshair.y / (float)h;
        float mark_y = vb_pos.y + norm_y * square_side;
        draw_list->AddLine(ImVec2(vb_pos.x, mark_y), ImVec2(vb_pos.x + depth_thickness, mark_y),
                           IM_COL32(255, 234, 0, 200), 2.0f);
    }
    ImGui::EndChild();

    // 3. Horizontal B-Scan (Width = square_side aligned strictly with Camera X)
    ImGui::BeginChild("HBScanBox", ImVec2(square_side + 12.0f, depth_thickness + 28.0f), true);
    ImGui::Text("H-BScan (Row Y=%d)", crosshair.y);
    ImVec2 hb_pos = ImGui::GetCursorScreenPos();
    ImVec2 hb_size(square_side, depth_thickness);

    if (hbscan_texture_) {
        ImGui::Image((__bridge void*)hbscan_texture_, hb_size);

        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        float norm_x = (float)crosshair.x / (float)w;
        float mark_x = hb_pos.x + norm_x * square_side;
        draw_list->AddLine(ImVec2(mark_x, hb_pos.y), ImVec2(mark_x, hb_pos.y + depth_thickness),
                           IM_COL32(0, 240, 255, 200), 2.0f);
    }
    ImGui::EndChild();

    ImGui::SameLine();

    // 4. Wide 1D Central A-Scan Depth Profile
    float ascan_width = std::max(220.0f, avail_w - square_side - 32.0f);
    ImGui::BeginChild("AScanBox", ImVec2(ascan_width, depth_thickness + 28.0f), true);
    ImGui::Text("1D A-Scan Profile (dB)");

    if (has_result_ && !latest_result_.ascan.empty()) {
        int depth_bins = (int)latest_result_.ascan.size();
        float peak_val = latest_result_.ascan[0];
        int peak_depth = 0;
        for (size_t i = 0; i < latest_result_.ascan.size(); ++i) {
            if (latest_result_.ascan[i] > peak_val) {
                peak_val = latest_result_.ascan[i];
                peak_depth = (int)i;
            }
        }
        ImGui::SameLine();
        ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "| Peak: %.1f dB @ z=%d", peak_val, peak_depth);

        if (ImPlot::BeginPlot("##AScanPlot", ImVec2(ascan_width - 16.0f, depth_thickness - 6.0f), ImPlotFlags_NoTitle | ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes("Depth z (bins)", "dB", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
            ImPlot::SetupAxesLimits(0, depth_bins, 20.0, 105.0, ImPlotCond_Always);
            ImPlot::PlotLine("Depth Profile", latest_result_.depth_axis.data(), latest_result_.ascan.data(), (int)latest_result_.ascan.size());
            ImPlot::EndPlot();
        }
    }
    ImGui::EndChild();
}

void AppGUI::render_volume3d_viewer() {
    float avail_w = ImGui::GetContentRegionAvail().x;
    float avail_h = ImGui::GetContentRegionAvail().y;

    ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Phase-OCT: 2D Unwrapped Map, Line Profile & 3D Topography Mesh");
    ImGui::Separator();

    // Controls Row: Range Sliders, ASM Refocusing, Immersion RI
    ImGui::BeginChild("PhaseParamBar", ImVec2(avail_w, 36), false);
    ImGui::SetNextItemWidth(130);
    if (ImGui::DragFloatRange2("Range", &phase_cmin_, &phase_cmax_, 0.1f, -25.0f, 25.0f, "Min: %.1f", "Max: %.1f")) {
        update_phase_textures();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90);
    if (ImGui::SliderFloat("dz [um]", &phase_dz_focus_um_, -25.0f, 25.0f, "%.1f")) {
        trigger_phase_reconstruction();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(75);
    if (ImGui::SliderFloat("RI", &phase_n_immersion_, 1.00f, 1.80f, "%.2f")) {
        trigger_phase_reconstruction();
    }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::SliderFloat("3D Height", &raymarch_z_stretch_, 1.0f, 12.0f, "%.1fx");
    ImGui::EndChild();

    float top_sub_h = (avail_h - 52.0f) * 0.58f;
    float bottom_sub_h = avail_h - 52.0f - top_sub_h - 10.0f;
    float half_w = (avail_w - 12.0f) * 0.5f;

    // -------------------------------------------------------------
    // TOP SUB-ROW: [ 2D Phase Map + Line Cut ] | [ 3D Topography Mesh ]
    // -------------------------------------------------------------
    // Left: 2D Phase Map
    ImGui::BeginChild("Phase2DBox", ImVec2(half_w, top_sub_h), true);
    ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "2D Unwrapped Phase (Drag mouse -> Section Line):");

    const float cb_w = 16.0f;
    const float cb_lbl_w = 55.0f;
    float max_img_size = std::min(half_w - 24.0f - cb_w - cb_lbl_w, top_sub_h - 32.0f);
    ImVec2 img_pos = ImGui::GetCursorScreenPos();

    if (phase_texture_) {
        ImGui::Image((__bridge void*)phase_texture_, ImVec2(max_img_size, max_img_size));

        // Interactive Line Drawing on Mouse Drag
        if (ImGui::IsItemHovered()) {
            ImVec2 m = ImGui::GetIO().MousePos;
            float norm_x = std::clamp((m.x - img_pos.x) / max_img_size, 0.0f, 1.0f);
            float norm_y = std::clamp((m.y - img_pos.y) / max_img_size, 0.0f, 1.0f);

            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                line_p0_ = ImVec2(norm_x, norm_y);
                line_p1_ = ImVec2(norm_x, norm_y);
                is_dragging_line_ = true;
            }
            if (is_dragging_line_ && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                line_p1_ = ImVec2(norm_x, norm_y);
            }
        }
        if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
            is_dragging_line_ = false;
        }

        // Render Line Overlay
        ImDrawList* draw_list = ImGui::GetWindowDrawList();
        ImVec2 s_p0 = ImVec2(img_pos.x + line_p0_.x * max_img_size, img_pos.y + line_p0_.y * max_img_size);
        ImVec2 s_p1 = ImVec2(img_pos.x + line_p1_.x * max_img_size, img_pos.y + line_p1_.y * max_img_size);

        draw_list->AddLine(s_p0, s_p1, IM_COL32(255, 255, 0, 255), 2.5f);
        draw_list->AddCircleFilled(s_p0, 4.0f, IM_COL32(0, 255, 255, 255));
        draw_list->AddCircleFilled(s_p1, 4.0f, IM_COL32(255, 0, 255, 255));
    }

    // Vertical Colorbar
    ImGui::SameLine();
    if (phase_colorbar_texture_) {
        ImGui::Image((__bridge void*)phase_colorbar_texture_, ImVec2(cb_w, max_img_size));
    }
    ImGui::SameLine();
    ImGui::BeginGroup();
    ImGui::Text("%.1f", phase_cmax_);
    ImGui::Dummy(ImVec2(0, max_img_size * 0.40f));
    ImGui::Text("%.1f", (phase_cmax_ + phase_cmin_) * 0.5f);
    ImGui::Dummy(ImVec2(0, max_img_size * 0.40f));
    ImGui::Text("%.1f", phase_cmin_);
    ImGui::EndGroup();

    ImGui::EndChild();

    ImGui::SameLine();

    // Right: 3D Topography Mesh / Point Cloud
    ImGui::BeginChild("Surface3DBox", ImVec2(half_w, top_sub_h), true);
    ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "3D Surface Topography (Rotate: Mouse, Zoom: Scroll):");

    const uint32_t* lut_jet = ColormapManager::get_lut(ColormapType::Jet);
    volume3d_engine_->render_raymarching(raymarch_yaw_, raymarch_pitch_, raymarch_zoom_, 0.0f, 1.0f, raymarch_z_stretch_, 0, lut_jet, 256);
    void* tex3d = volume3d_engine_->get_rendered_texture_ptr();
    if (tex3d) {
        ImGui::Image(tex3d, ImVec2(max_img_size, max_img_size));
        if (ImGui::IsItemHovered()) {
            if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                raymarch_yaw_ += ImGui::GetIO().MouseDelta.x * 0.008f;
                raymarch_pitch_ += ImGui::GetIO().MouseDelta.y * 0.008f;
            }
            if (ImGui::GetIO().MouseWheel != 0.0f) {
                raymarch_zoom_ = std::clamp(raymarch_zoom_ - ImGui::GetIO().MouseWheel * 0.15f, 0.4f, 6.0f);
            }
        }
    }
    ImGui::EndChild();

    // -------------------------------------------------------------
    // BOTTOM SUB-ROW: Dynamic 1D Phase Profile along Drawn Line (Dynamic Y-Axis Autoscale)
    // -------------------------------------------------------------
    ImGui::BeginChild("LineProfileBox", ImVec2(avail_w, bottom_sub_h), true);
    ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "1D Dynamic Phase Profile along Section Line [P0 -> P1] (Y-Autoscaling):");

    const auto& ph = volume3d_engine_->get_phase_reconstructed();
    int w = processor_->get_width();
    int h = processor_->get_height();

    if (!ph.empty() && w > 0 && h > 0) {
        const int num_samples = 300;
        std::vector<double> dist_axis(num_samples);
        std::vector<double> phase_vals(num_samples);

        float x0 = line_p0_.x * (float)(w - 1);
        float y0 = line_p0_.y * (float)(h - 1);
        float x1 = line_p1_.x * (float)(w - 1);
        float y1 = line_p1_.y * (float)(h - 1);

        double total_dist_um = std::hypot((double)(x1 - x0), (double)(y1 - y0)) * (double)phase_dx_um_;

        double p_line_min = 1e9;
        double p_line_max = -1e9;

        for (int i = 0; i < num_samples; ++i) {
            float t = (float)i / (float)(num_samples - 1);
            float sx = x0 + t * (x1 - x0);
            float sy = y0 + t * (y1 - y0);

            int ix = std::clamp((int)std::round(sx), 0, w - 1);
            int iy = std::clamp((int)std::round(sy), 0, h - 1);

            dist_axis[i] = t * total_dist_um;
            double v = (double)ph[(size_t)iy * w + ix];
            phase_vals[i] = v;
            if (v < p_line_min) p_line_min = v;
            if (v > p_line_max) p_line_max = v;
        }

        double margin = std::max(0.2, (p_line_max - p_line_min) * 0.1);

        if (ImPlot::BeginPlot("##PhaseLinePlot", ImVec2(avail_w - 20.0f, bottom_sub_h - 32.0f), ImPlotFlags_NoTitle | ImPlotFlags_NoLegend)) {
            ImPlot::SetupAxes("Physical distance on sample (um)", "Phase (rad)", ImPlotAxisFlags_None, ImPlotAxisFlags_None);
            ImPlot::SetupAxesLimits(0.0, std::max(1.0, total_dist_um), p_line_min - margin, p_line_max + margin, ImPlotCond_Always);
            ImPlot::PlotLine("Profil Fazy", dist_axis.data(), phase_vals.data(), num_samples);
            ImPlot::EndPlot();
        }
    }
    ImGui::EndChild();
}

void AppGUI::render_control_sidebar() {
}

void AppGUI::render_top_bar() {
}

} // namespace oct

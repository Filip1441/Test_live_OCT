#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <fstream>
#include <zlib.h>
#include <hdf5.h>

#include "imgui.h"
#include "imgui_impl_metal.h"
#include "imgui_impl_osx.h"
#include "implot.h"

#include "hdf5_loader.hpp"
#include "volume3d_engine.hpp"
#include "colormaps.hpp"
#include "file_dialog.hpp"

namespace {

float corr2(const float* a, const float* b, size_t n) {
    double sum_a = 0.0, sum_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        sum_a += a[i];
        sum_b += b[i];
    }
    double mean_a = sum_a / (double)n;
    double mean_b = sum_b / (double)n;

    double num = 0.0, denom_a = 0.0, denom_b = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double da = (double)a[i] - mean_a;
        double db = (double)b[i] - mean_b;
        num += da * db;
        denom_a += da * da;
        denom_b += db * db;
    }
    double denom = std::sqrt(denom_a * denom_b);
    return (denom > 1e-12) ? (float)(num / denom) : 0.0f;
}

std::vector<float> apply_transform(const float* src, int w, int h, int tf_idx) {
    std::vector<float> dst((size_t)w * h);
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            int src_x = x;
            int src_y = y;
            switch (tf_idx) {
                case 0: // Identity
                    src_x = x; src_y = y; break;
                case 1: // fliplr
                    src_x = w - 1 - x; src_y = y; break;
                case 2: // flipud
                    src_x = x; src_y = h - 1 - y; break;
                case 3: // fliplr(flipud)
                    src_x = w - 1 - x; src_y = h - 1 - y; break;
                case 4: // Transpose
                    src_x = y; src_y = x; break;
                case 5: // fliplr(Transpose)
                    src_x = w - 1 - y; src_y = x; break;
                case 6: // flipud(Transpose)
                    src_x = y; src_y = h - 1 - x; break;
                case 7: // fliplr(flipud(Transpose))
                    src_x = w - 1 - y; src_y = h - 1 - x; break;
            }
            if (src_x >= 0 && src_x < w && src_y >= 0 && src_y < h) {
                dst[(size_t)y * w + x] = src[(size_t)src_y * w + src_x];
            }
        }
    }
    return dst;
}

bool load_mat_v5(const std::string& mat_path, std::vector<float>& out_phase, int& out_w, int& out_h) {
    std::ifstream file(mat_path, std::ios::binary);
    if (!file.is_open()) return false;

    char header[128];
    file.read(header, 128);
    if (file.gcount() < 128) return false;

    std::string header_str(header, 128);
    if (header_str.find("MATLAB 5.0") == std::string::npos) return false;

    while (file) {
        uint32_t tag[2] = {0, 0};
        file.read(reinterpret_cast<char*>(tag), 8);
        if (file.gcount() < 8) break;

        uint32_t dtype = tag[0];
        uint32_t nbytes = tag[1];
        if (dtype == 0 || nbytes == 0) break;

        if (dtype == 15) { // miCOMPRESSED
            std::vector<uint8_t> comp(nbytes);
            file.read(reinterpret_cast<char*>(comp.data()), nbytes);

            // Decompress with zlib inflate
            std::vector<uint8_t> decomp(nbytes * 4 + 65536);
            z_stream strm = {};
            strm.next_in = comp.data();
            strm.avail_in = (uInt)comp.size();

            if (inflateInit(&strm) != Z_OK) continue;

            while (true) {
                strm.next_out = decomp.data() + strm.total_out;
                strm.avail_out = (uInt)(decomp.size() - strm.total_out);

                int ret = inflate(&strm, Z_NO_FLUSH);
                if (ret == Z_STREAM_END) break;
                if (ret != Z_OK) {
                    inflateEnd(&strm);
                    break;
                }
                if (strm.avail_out == 0) {
                    decomp.resize(decomp.size() * 2);
                }
            }
            inflateEnd(&strm);

            // Parse miMATRIX inside decomp
            if (decomp.size() < 32) continue;
            size_t offset = 0;
            uint32_t mat_type = *reinterpret_cast<uint32_t*>(&decomp[offset]);
            if (mat_type != 14) continue; // miMATRIX
            offset += 8;

            // 1. Array Flags
            uint32_t flags_len = *reinterpret_cast<uint32_t*>(&decomp[offset + 4]);
            offset += 8 + ((flags_len + 7) & ~7);

            // 2. Dimensions Array
            uint32_t dims_len = *reinterpret_cast<uint32_t*>(&decomp[offset + 4]);
            offset += 8;
            int32_t d0 = *reinterpret_cast<int32_t*>(&decomp[offset]);
            int32_t d1 = *reinterpret_cast<int32_t*>(&decomp[offset + 4]);
            offset += ((dims_len + 7) & ~7);

            out_h = d0;
            out_w = d1;

            // 3. Array Name
            uint32_t name_tag0 = *reinterpret_cast<uint32_t*>(&decomp[offset]);
            if (name_tag0 >> 16) {
                offset += 8;
            } else {
                uint32_t name_len = *reinterpret_cast<uint32_t*>(&decomp[offset + 4]);
                offset += 8 + ((name_len + 7) & ~7);
            }

            // 4. Real Part Data
            uint32_t data_type = *reinterpret_cast<uint32_t*>(&decomp[offset]);
            offset += 8;

            size_t total = (size_t)out_w * out_h;
            out_phase.resize(total);

            if (data_type == 7) { // miSINGLE
                const float* ptr = reinterpret_cast<const float*>(&decomp[offset]);
                for (int y = 0; y < out_h; ++y) {
                    for (int x = 0; x < out_w; ++x) {
                        out_phase[(size_t)y * out_w + x] = ptr[(size_t)x * out_h + y];
                    }
                }
                return true;
            } else if (data_type == 9) { // miDOUBLE
                const double* ptr = reinterpret_cast<const double*>(&decomp[offset]);
                for (int y = 0; y < out_h; ++y) {
                    for (int x = 0; x < out_w; ++x) {
                        out_phase[(size_t)y * out_w + x] = (float)ptr[(size_t)x * out_h + y];
                    }
                }
                return true;
            }
        } else {
            file.seekg(nbytes, std::ios::cur);
        }
    }
    return false;
}

bool load_mat_target(const std::string& mat_path, std::vector<float>& out_phase, int& out_w, int& out_h) {
    if (load_mat_v5(mat_path, out_phase, out_w, out_h)) {
        return true;
    }

    hid_t file_id = H5Fopen(mat_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) return false;

    std::string candidate_names[] = {"ph_reconstructed", "ph_initial", "phase", "ph", "data"};
    std::string found_name = "";
    for (const auto& name : candidate_names) {
        if (H5Lexists(file_id, name.c_str(), H5P_DEFAULT) > 0) {
            found_name = name;
            break;
        }
    }

    if (found_name.empty()) {
        H5Fclose(file_id);
        return false;
    }

    hid_t ds_id = H5Dopen2(file_id, found_name.c_str(), H5P_DEFAULT);
    if (ds_id < 0) {
        H5Fclose(file_id);
        return false;
    }

    hid_t filespace = H5Dget_space(ds_id);
    hsize_t dims[2];
    H5Sget_simple_extent_dims(filespace, dims, NULL);
    out_w = (int)dims[0];
    out_h = (int)dims[1];

    size_t total = (size_t)out_w * out_h;
    std::vector<float> col_major(total);
    H5Dread(ds_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, col_major.data());

    H5Sclose(filespace);
    H5Dclose(ds_id);
    H5Fclose(file_id);

    // Transpose from MATLAB column-major to C++ row-major
    out_phase.resize(total);
    for (int y = 0; y < out_h; ++y) {
        for (int x = 0; x < out_w; ++x) {
            out_phase[(size_t)y * out_w + x] = col_major[(size_t)x * out_h + y];
        }
    }
    return true;
}

id<MTLTexture> create_texture_from_floats(
    id<MTLDevice> device,
    const float* data,
    int w,
    int h,
    float cmin,
    float cmax,
    const uint32_t* lut
) {
    if (!device || !data || w <= 0 || h <= 0) return nil;

    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];

    std::vector<uint32_t> rgba((size_t)w * h);
    float range = (cmax > cmin) ? (cmax - cmin) : 1.0f;

    for (size_t i = 0; i < (size_t)w * h; ++i) {
        float norm = std::clamp((data[i] - cmin) / range, 0.0f, 1.0f);
        int idx = std::clamp((int)(norm * 255.0f), 0, 255);
        rgba[i] = lut ? lut[idx] : 0xFFFFFFFF;
    }

    MTLRegion region = MTLRegionMake2D(0, 0, w, h);
    [tex replaceRegion:region mipmapLevel:0 withBytes:rgba.data() bytesPerRow:w * sizeof(uint32_t)];
    return tex;
}

} // anonymous namespace

@interface ValidateAppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation ValidateAppDelegate
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}
@end

@interface ValidateCustomView : NSView
@end

@implementation ValidateCustomView
- (BOOL)acceptsFirstResponder { return YES; }
@end

int main(int argc, const char * argv[]) {
    // 0. Prompt for Refractive Index (RI) in Terminal
    float n_immersion = 1.52f;
    std::cout << "\n=================================================================" << std::endl;
    std::cout << "   Phase-OCT Pure C++ Ultra-Low RAM Validation Visualizer         " << std::endl;
    std::cout << "=================================================================" << std::endl;
    std::cout << "Podaj wspolczynnik zalamania medium immersyjnego (RI) [domyslnie 1.52]: ";
    std::cout.flush();

    std::string ri_input;
    if (std::getline(std::cin, ri_input)) {
        ri_input.erase(std::remove_if(ri_input.begin(), ri_input.end(), ::isspace), ri_input.end());
        if (!ri_input.empty()) {
            try {
                n_immersion = std::stof(ri_input);
            } catch (...) {
                n_immersion = 1.52f;
            }
        }
    }
    std::cout << "--> Wybrany wspolczynnik zalamania (RI): " << n_immersion << std::endl;

    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        ValidateAppDelegate* delegate = [[ValidateAppDelegate alloc] init];
        [NSApp setDelegate:delegate];

        // 1. Create macOS Window
        NSRect frame = NSMakeRect(50, 50, 1680, 1020);
        NSUInteger styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                               NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable;

        NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:styleMask
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        [window setTitle:@"Unwrapped Phase-OCT Engine [C++ MATLAB-Validation Visualizer]"];
        [window center];

        // 2. Setup Metal Device
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) return 1;

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        CAMetalLayer* metalLayer = [CAMetalLayer layer];
        metalLayer.device = device;
        metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;

        ValidateCustomView* contentView = [[ValidateCustomView alloc] initWithFrame:frame];
        [contentView setWantsLayer:YES];
        [contentView setLayer:metalLayer];
        [window setContentView:contentView];
        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        // 3. Initialize ImGui & ImPlot
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
        ImGui::StyleColorsDark();

        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 6.0f;
        style.FrameRounding = 4.0f;
        style.ItemSpacing = ImVec2(8, 6);
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.08f, 0.11f, 1.00f);
        style.Colors[ImGuiCol_ChildBg] = ImVec4(0.09f, 0.11f, 0.15f, 1.00f);

        ImGui_ImplOSX_Init(contentView);
        ImGui_ImplMetal_Init(device);

        // Paths
        std::string obj_path = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.hdf5";
        std::string ref_path = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/reference___2026-8-26_16-48-32.hdf5";
        std::string mat_path = "/Users/filip/Documents/kod_matlab/OCT_test_rekonstrukcji/moj_wynik.mat";

        if (argc > 1) obj_path = argv[1];
        if (argc > 2) ref_path = argv[2];
        if (argc > 3) mat_path = argv[3];

        // Load datasets
        std::cout << "[Validation Visualizer] Loading Object HDF5: " << obj_path << std::endl;
        auto ds_obj = oct::Hdf5Loader::load(obj_path);

        std::cout << "[Validation Visualizer] Loading Reference HDF5: " << ref_path << std::endl;
        auto ds_ref = oct::Hdf5Loader::load(ref_path);

        std::vector<float> target_phase;
        int target_w = 0, target_h = 0;
        bool has_target = load_mat_target(mat_path, target_phase, target_w, target_h);

        // Run Phase-OCT Reconstruction
        auto engine = std::make_unique<oct::Volume3DEngine>(device, ds_obj->width, ds_obj->height, ds_obj->num_wavelengths);
        oct::PhaseOctStats stats;
        engine->reconstruct_phase_oct(
            ds_obj->data.data(),
            ds_ref->data.data(),
            ds_obj->wavelengths_nm,
            0.0f, n_immersion, 0.5f, false,
            oct::PipelineMode::Optimized_DTFT,
            oct::ComputeBackend3D::CPUMultiCore,
            stats
        );

        // Metrics & transforms
        const auto& ph_recon = engine->get_phase_reconstructed();
        size_t total_pix = (size_t)ds_obj->width * ds_obj->height;

        int best_tf_idx = 4; // Default #5 Transpose
        float best_sign = 1.0f;
        float best_corr = 0.0f;
        std::vector<float> aligned_target;
        std::vector<float> diff_phase(total_pix, 0.0f);
        float rmse_val = 0.0f;

        if (has_target) {
            for (int i = 0; i < 8; ++i) {
                std::vector<float> test_tgt = apply_transform(target_phase.data(), ds_obj->width, ds_obj->height, i);
                float c = corr2(ph_recon.data(), test_tgt.data(), total_pix);
                if (std::abs(c) > best_corr) {
                    best_corr = std::abs(c);
                    best_tf_idx = i;
                    best_sign = (c >= 0.0f) ? 1.0f : -1.0f;
                }
            }

            aligned_target = apply_transform(target_phase.data(), ds_obj->width, ds_obj->height, best_tf_idx);

            // Align background
            int y_mid0 = (int)(ds_obj->height * 0.25f);
            int y_mid1 = (int)(ds_obj->height * 0.75f);
            int x_mid0 = (int)(ds_obj->width * 0.25f);
            int x_mid1 = (int)(ds_obj->width * 0.75f);

            double sum_tgt_bg = 0.0, sum_rec_bg = 0.0;
            int count_bg = 0;
            for (int y = 0; y < ds_obj->height; ++y) {
                for (int x = 0; x < ds_obj->width; ++x) {
                    if (y >= y_mid0 && y <= y_mid1 && x >= x_mid0 && x <= x_mid1) continue;
                    size_t idx = (size_t)y * ds_obj->width + x;
                    sum_tgt_bg += aligned_target[idx];
                    sum_rec_bg += (double)ph_recon[idx] * best_sign;
                    count_bg++;
                }
            }
            double mean_tgt_bg = sum_tgt_bg / (double)count_bg;
            double mean_rec_bg = sum_rec_bg / (double)count_bg;

            double sum_sq = 0.0;
            for (size_t i = 0; i < total_pix; ++i) {
                float rec_val = (float)((double)ph_recon[i] * best_sign - mean_rec_bg + mean_tgt_bg);
                diff_phase[i] = rec_val - aligned_target[i];
                sum_sq += (double)diff_phase[i] * (double)diff_phase[i];
            }
            rmse_val = (float)std::sqrt(sum_sq / (double)total_pix);

            std::cout << "=================================================================" << std::endl;
            std::cout << "[Validation Visualizer] Ground Truth MAT Comparison: " << mat_path << std::endl;
            std::cout << "  - Best Transform:     #" << (best_tf_idx + 1) << std::endl;
            std::cout << "  - 2D Correlation:     " << std::fixed << std::setprecision(5) << best_corr * 100.0f << " %" << std::endl;
            std::cout << "  - Phase RMSE:         " << std::setprecision(4) << rmse_val << " rad" << std::endl;
            std::cout << "  - Step Height (C++):  " << std::setprecision(3) << stats.step_height_um << " um" << std::endl;
            std::cout << "=================================================================" << std::endl;
        }

        // Textures for three subplots
        const uint32_t* lut_jet = oct::ColormapManager::get_lut(oct::ColormapType::Jet);
        id<MTLTexture> tex_recon = create_texture_from_floats(device, ph_recon.data(), ds_obj->width, ds_obj->height, -4.0f, 3.0f, lut_jet);
        id<MTLTexture> tex_target = has_target ? create_texture_from_floats(device, aligned_target.data(), ds_obj->width, ds_obj->height, -4.0f, 3.0f, lut_jet) : nil;
        id<MTLTexture> tex_diff = has_target ? create_texture_from_floats(device, diff_phase.data(), ds_obj->width, ds_obj->height, -1.0f, 1.0f, lut_jet) : nil;

        // Interactive states
        float yaw = 0.75f, pitch = 1.25f, zoom = 1.8f, z_scale = 3.5f;
        int active_cmap_idx = 0; // 0=Jet
        float cmin = -4.0f, cmax = 3.0f;
        float diff_cmin = -1.0f, diff_cmax = 1.0f;
        bool invert_phase = false;

        MTLRenderPassDescriptor* renderPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];

        // 4. Render Loop
        while ([window isVisible]) {
            @autoreleasepool {
                NSEvent* event;
                while ((event = [NSApp nextEventMatchingMask:NSEventMaskAny
                                                   untilDate:[NSDate distantPast]
                                                      inMode:NSDefaultRunLoopMode
                                                     dequeue:YES])) {
                    [NSApp sendEvent:event];
                    [NSApp updateWindows];
                }

                if (![window isVisible]) break;

                NSRect bounds = [contentView bounds];
                if (bounds.size.width <= 0 || bounds.size.height <= 0) {
                    [NSThread sleepForTimeInterval:0.01];
                    continue;
                }

                metalLayer.drawableSize = CGSizeMake(bounds.size.width * [window backingScaleFactor],
                                                     bounds.size.height * [window backingScaleFactor]);
                id<CAMetalDrawable> drawable = [metalLayer nextDrawable];
                if (!drawable) continue;

                renderPassDesc.colorAttachments[0].texture = drawable.texture;
                renderPassDesc.colorAttachments[0].loadAction = MTLLoadActionClear;
                renderPassDesc.colorAttachments[0].clearColor = MTLClearColorMake(0.04, 0.05, 0.07, 1.0);
                renderPassDesc.colorAttachments[0].storeAction = MTLStoreActionStore;

                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDesc];

                ImGui_ImplMetal_NewFrame(renderPassDesc);
                ImGui_ImplOSX_NewFrame(contentView);
                ImGui::NewFrame();

                // UI Layout
                ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(bounds.size.width - 20, bounds.size.height - 20), ImGuiCond_Always);

                ImGui::Begin("Phase-OCT Validation & Optical Topography Dashboard", nullptr,
                             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse | ImGuiWindowFlags_MenuBar);

                // Menu Bar
                if (ImGui::BeginMenuBar()) {
                    if (ImGui::BeginMenu("File")) {
                        if (ImGui::MenuItem("Load Object HDF5...")) {
                            std::string p = oct::FileDialog::open_file("Select Object HDF5");
                            if (!p.empty()) {
                                ds_obj = oct::Hdf5Loader::load(p);
                                engine->reconstruct_phase_oct(ds_obj->data.data(), ds_ref->data.data(), ds_obj->wavelengths_nm, 0.0f, 1.52f, 0.5f, false, oct::PipelineMode::Optimized_DTFT, oct::ComputeBackend3D::CPUMultiCore, stats);
                            }
                        }
                        if (ImGui::MenuItem("Load Reference HDF5...")) {
                            std::string p = oct::FileDialog::open_file("Select Reference HDF5");
                            if (!p.empty()) {
                                ds_ref = oct::Hdf5Loader::load(p);
                                engine->reconstruct_phase_oct(ds_obj->data.data(), ds_ref->data.data(), ds_obj->wavelengths_nm, 0.0f, 1.52f, 0.5f, false, oct::PipelineMode::Optimized_DTFT, oct::ComputeBackend3D::CPUMultiCore, stats);
                            }
                        }
                        if (ImGui::MenuItem("Close Visualizer")) [window close];
                        ImGui::EndMenu();
                    }
                    ImGui::EndMenuBar();
                }

                // Header Metrics Banner
                ImGui::BeginChild("Banner", ImVec2(0, 75), true);
                ImGui::Columns(4, "BannerCols", false);

                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Reconstruction Time:");
                ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "%.1f ms (%.2f s)", stats.compute_time_ms, stats.compute_time_ms / 1000.0f);
                ImGui::Text("z_obj: %d | z_ref: %d", stats.z_obj_index, stats.z_ref_index);

                ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Ground Truth Match:");
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Correlation: %.2f %%", best_corr * 100.0f);
                ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.3f, 1.0f), "Phase RMSE:  %.3f rad", rmse_val);

                ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Physical Step Topography:");
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Step Height: %.3f um", stats.step_height_um);
                ImGui::Text("Phase Range: [%.2f .. %.2f] rad", stats.phase_min_rad, stats.phase_max_rad);

                ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Controls & Alignment:");
                const char* tf_names[] = {"Identity", "fliplr", "flipud", "fliplr(flipud)", "Transpose (#5)", "fliplr(Trans)", "flipud(Trans)", "fliplr(flipud(Trans))"};
                if (ImGui::Combo("##Tf", &best_tf_idx, tf_names, 8)) {
                    aligned_target = apply_transform(target_phase.data(), ds_obj->width, ds_obj->height, best_tf_idx);
                    tex_target = create_texture_from_floats(device, aligned_target.data(), ds_obj->width, ds_obj->height, cmin, cmax, lut_jet);
                }
                if (ImGui::Checkbox("Invert Phase Sign (* -1)", &invert_phase)) {
                    std::vector<float> ph_inv(total_pix);
                    float sign_mult = invert_phase ? -1.0f : 1.0f;
                    for (size_t i = 0; i < total_pix; ++i) ph_inv[i] = ph_recon[i] * sign_mult;
                    tex_recon = create_texture_from_floats(device, ph_inv.data(), ds_obj->width, ds_obj->height, cmin, cmax, lut_jet);

                    if (has_target) {
                        for (size_t i = 0; i < total_pix; ++i) {
                            diff_phase[i] = ph_inv[i] - aligned_target[i];
                        }
                        tex_diff = create_texture_from_floats(device, diff_phase.data(), ds_obj->width, ds_obj->height, -1.0f, 1.0f, lut_jet);
                    }
                }
                ImGui::Columns(1);
                ImGui::EndChild();

                // 3 Comparison Subplots (Side-by-Side just like MATLAB)
                float avail_w = ImGui::GetContentRegionAvail().x;
                float avail_h = ImGui::GetContentRegionAvail().y;
                float col_w = (avail_w - 24.0f) / 3.0f;
                float plot_h = std::min(col_w, avail_h - 220.0f);

                // Subplot 1: C++ Reconstructed Phase
                ImGui::BeginChild("Subplot1", ImVec2(col_w, plot_h + 35.0f), true);
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "1. C++ Reconstruction (Unwrapped)");
                if (tex_recon) {
                    ImGui::Image((__bridge void*)tex_recon, ImVec2(col_w - 16.0f, plot_h));
                }
                ImGui::EndChild();

                ImGui::SameLine();

                // Subplot 2: Ground Truth Target Result
                ImGui::BeginChild("Subplot2", ImVec2(col_w, plot_h + 35.0f), true);
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "2. Target Result (MATLAB Ground Truth)");
                if (tex_target) {
                    ImGui::Image((__bridge void*)tex_target, ImVec2(col_w - 16.0f, plot_h));
                }
                ImGui::EndChild();

                ImGui::SameLine();

                // Subplot 3: Absolute Phase Difference Map
                ImGui::BeginChild("Subplot3", ImVec2(col_w, plot_h + 35.0f), true);
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "3. Phase Difference (RMSE: %.3f rad)", rmse_val);
                if (tex_diff) {
                    ImGui::Image((__bridge void*)tex_diff, ImVec2(col_w - 16.0f, plot_h));
                }
                ImGui::EndChild();

                // Bottom Row: 3D Surface Topography & 1D Cross-Section
                ImGui::BeginChild("BottomRow", ImVec2(avail_w, avail_h - plot_h - 45.0f), true);
                ImGui::Columns(2, "BottomCols", false);

                // 3D Topography Mesh Surface
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "3D Optical Topography Surface [Drag Mouse to Orbit, Scroll to Zoom]:");
                engine->render_raymarching(yaw, pitch, zoom, 0.0f, 1.0f, z_scale, 0, lut_jet, 256);
                void* tex3d = engine->get_rendered_texture_ptr();
                if (tex3d) {
                    float s_size = std::min(avail_w * 0.45f, avail_h - plot_h - 90.0f);
                    ImGui::Image(tex3d, ImVec2(s_size, s_size));
                    if (ImGui::IsItemHovered()) {
                        if (ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                            yaw += io.MouseDelta.x * 0.008f;
                            pitch = std::clamp(pitch - io.MouseDelta.y * 0.008f, 0.1f, 3.04f);
                        }
                        if (io.MouseWheel != 0.0f) {
                            zoom = std::clamp(zoom - io.MouseWheel * 0.15f, 0.5f, 5.0f);
                        }
                    }
                }

                ImGui::NextColumn();

                // 1D Center Cross-Section Cut
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Central 1D Step Profile Cross-Section (Row Y = %d):", ds_obj->height / 2);
                int cy_mid = ds_obj->height / 2;
                std::vector<float> rec_profile(ds_obj->width);
                std::vector<float> tgt_profile(ds_obj->width);
                std::vector<float> x_axis(ds_obj->width);
                float prof_sign = invert_phase ? -1.0f : 1.0f;
                for (int x = 0; x < ds_obj->width; ++x) {
                    rec_profile[x] = ph_recon[(size_t)cy_mid * ds_obj->width + x] * prof_sign;
                    tgt_profile[x] = aligned_target.empty() ? 0.0f : aligned_target[(size_t)cy_mid * ds_obj->width + x];
                    x_axis[x] = (float)x * 0.5f; // um
                }

                if (ImPlot::BeginPlot("##ProfilePlot", ImVec2(col_w * 1.4f, avail_h - plot_h - 90.0f))) {
                    ImPlot::SetupAxes("Position x [um]", "Phase Delay [rad]");
                    ImPlot::PlotLine("C++ Reconstructed", x_axis.data(), rec_profile.data(), ds_obj->width);
                    if (!aligned_target.empty()) {
                        ImPlot::PlotLine("Target Ground Truth", x_axis.data(), tgt_profile.data(), ds_obj->width);
                    }
                    ImPlot::EndPlot();
                }

                ImGui::Columns(1);
                ImGui::EndChild();

                ImGui::End();

                ImGui::Render();
                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);

                [renderEncoder endEncoding];
                [commandBuffer presentDrawable:drawable];
                [commandBuffer commit];
            }
        }

        ImGui_ImplMetal_Shutdown();
        ImGui_ImplOSX_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
    return 0;
}

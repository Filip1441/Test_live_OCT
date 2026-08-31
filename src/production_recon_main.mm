#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#import <ImageIO/ImageIO.h>
#import <CoreGraphics/CoreGraphics.h>
#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <filesystem>

#include "imgui.h"
#include "imgui_impl_metal.h"
#include "imgui_impl_osx.h"
#include "implot.h"

#include "hdf5_loader.hpp"
#include "volume3d_engine.hpp"
#include "colormaps.hpp"
#include "file_dialog.hpp"

namespace {

// Export Standard Level 5 MATLAB *.mat File (100% compatible with MATLAB load())
void export_mat5_matrix(const std::string& filepath, const float* data, int width, int height) {
    std::ofstream out(filepath, std::ios::binary);
    if (!out.is_open()) return;

    // 128-byte Header
    std::string header_text = "MATLAB 5.0 MAT-file, Platform: MACA64, Created by Phase-OCT C++ High-Speed Engine";
    header_text.resize(116, ' ');
    out.write(header_text.data(), 116);

    uint64_t subsystem_offset = 0;
    uint16_t version = 0x0100;
    uint16_t endian = 0x4D49; // 'MI'
    out.write(reinterpret_cast<const char*>(&subsystem_offset), 8);
    out.write(reinterpret_cast<const char*>(&version), 2);
    out.write(reinterpret_cast<const char*>(&endian), 2);

    // Matrix Elements Payload
    std::vector<uint8_t> payload;

    auto append_tag = [&](uint32_t type, uint32_t nbytes, const void* src) {
        uint32_t tag[2] = {type, nbytes};
        size_t cur = payload.size();
        payload.resize(cur + 8 + nbytes);
        std::memcpy(payload.data() + cur, tag, 8);
        if (src && nbytes > 0) {
            std::memcpy(payload.data() + cur + 8, src, nbytes);
        }
        // 8-byte padding
        size_t pad = (8 - (nbytes % 8)) % 8;
        if (pad > 0) payload.insert(payload.end(), pad, 0);
    };

    // 1. Array Flags: mxSINGLE_CLASS = 7
    uint32_t flags[2] = {7, 0};
    append_tag(6, 8, flags); // miUINT32

    // 2. Dimensions Array: [height, width]
    int32_t dims[2] = {height, width};
    append_tag(5, 8, dims); // miINT32

    // 3. Array Name: "ph_reconstructed"
    std::string name = "ph_reconstructed";
    append_tag(1, (uint32_t)name.size(), name.data()); // miINT8

    // 4. Real Data (Column-major transpose: MATLAB stores columns contiguous)
    size_t total = (size_t)width * height;
    std::vector<float> col_major(total);
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            col_major[(size_t)x * height + y] = data[(size_t)y * width + x];
        }
    }
    append_tag(7, (uint32_t)(total * sizeof(float)), col_major.data()); // miSINGLE

    // Write miMATRIX (type 14) tag
    uint32_t mat_tag[2] = {14, (uint32_t)payload.size()};
    out.write(reinterpret_cast<const char*>(mat_tag), 8);
    out.write(reinterpret_cast<const char*>(payload.data()), payload.size());

    std::cout << "[Export] Zapisano plik MATLAB: " << filepath << " (zmienna: ph_reconstructed [" 
              << height << "x" << width << "])" << std::endl;
}

// Export High-Resolution PNG with Vertical Jet Colorbar Strip
void export_png_with_colorbar(
    const std::string& filepath,
    const float* phase_data,
    int width,
    int height,
    float phase_min,
    float phase_max,
    const uint32_t* lut_jet
) {
    @autoreleasepool {
        const int cb_margin = 16;
        const int cb_width = 36;
        const int total_w = width + cb_margin + cb_width + 12;
        const int total_h = height;

        std::vector<uint32_t> rgba_total((size_t)total_w * total_h, 0xFF141A22); // dark background

        float p_range = (phase_max > phase_min) ? (phase_max - phase_min) : 1.0f;

        // 1. Render 2D Phase Map
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float val = phase_data[(size_t)y * width + x];
                float norm = std::clamp((val - phase_min) / p_range, 0.0f, 1.0f);
                int idx = std::clamp((int)(norm * 255.0f), 0, 255);
                rgba_total[(size_t)y * total_w + x] = lut_jet[idx];
            }
        }

        // 2. Render Vertical Jet Colorbar Strip (Top = Max, Bottom = Min)
        int cb_start_x = width + cb_margin;
        for (int y = 0; y < height; ++y) {
            float norm = 1.0f - ((float)y / (float)(height - 1)); // 1.0 at top, 0.0 at bottom
            int idx = std::clamp((int)(norm * 255.0f), 0, 255);
            uint32_t color = lut_jet[idx];

            for (int cx = 0; cx < cb_width; ++cx) {
                rgba_total[(size_t)y * total_w + (cb_start_x + cx)] = color;
            }
        }

        CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
        CGContextRef context = CGBitmapContextCreate(
            (void*)rgba_total.data(), total_w, total_h, 8, total_w * 4, colorSpace,
            kCGImageAlphaNoneSkipLast | kCGBitmapByteOrder32Little
        );
        CGImageRef imageRef = CGBitmapContextCreateImage(context);

        NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:filepath.c_str()]];
        CGImageDestinationRef dest = CGImageDestinationCreateWithURL((__bridge CFURLRef)url, CFSTR("public.png"), 1, NULL);
        if (dest && imageRef) {
            CGImageDestinationAddImage(dest, imageRef, NULL);
            CGImageDestinationFinalize(dest);
            CFRelease(dest);
        }
        if (imageRef) CGImageRelease(imageRef);
        if (context) CGContextRelease(context);
        if (colorSpace) CGColorSpaceRelease(colorSpace);
        std::cout << "[Export] Zapisano obraz PNG z Colorbarem: " << filepath << " [" << total_w << "x" << total_h << "]" << std::endl;
    }
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

id<MTLTexture> create_colorbar_texture(id<MTLDevice> device, const uint32_t* lut, int lut_size = 256) {
    if (!device || !lut) return nil;
    int w = 1, h = lut_size;
    MTLTextureDescriptor* desc = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                                                    width:w
                                                                                   height:h
                                                                                mipmapped:NO];
    desc.usage = MTLTextureUsageShaderRead;
    id<MTLTexture> tex = [device newTextureWithDescriptor:desc];

    // Invert vertically so index 255 (red) is top, index 0 (blue) is bottom
    std::vector<uint32_t> cb_pixels(h);
    for (int y = 0; y < h; ++y) {
        cb_pixels[y] = lut[255 - y];
    }
    MTLRegion region = MTLRegionMake2D(0, 0, w, h);
    [tex replaceRegion:region mipmapLevel:0 withBytes:cb_pixels.data() bytesPerRow:w * sizeof(uint32_t)];
    return tex;
}

} // anonymous namespace

@interface ReconAppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation ReconAppDelegate
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}
@end

@interface ReconCustomView : NSView
@end

@implementation ReconCustomView
- (BOOL)acceptsFirstResponder { return YES; }
@end

int main(int argc, const char * argv[]) {
    std::cout << "=================================================================" << std::endl;
    std::cout << "   Phase-OCT Production High-Speed Reconstruction Engine          " << std::endl;
    std::cout << "=================================================================" << std::endl;

    // 0a. Execution Mode Selection
    int exec_mode = 1;
    std::cout << "Select operation mode:\n"
              << "  [1] Optimized C++ with GUI (High speed + Miguel 2D)\n"
              << "  [2] Optimized C++ headless (Headless + Miguel 2D)\n"
              << "  [3] MATLAB Exact with GUI (Max precision, slower)\n"
              << "  [4] MATLAB Exact headless (Headless)\n"
              << "Select mode [1-4, default 1]: ";
    std::cout.flush();

    std::string mode_input;
    if (std::getline(std::cin, mode_input)) {
        mode_input.erase(std::remove_if(mode_input.begin(), mode_input.end(), ::isspace), mode_input.end());
        if (!mode_input.empty()) {
            try {
                exec_mode = std::clamp(std::stoi(mode_input), 1, 4);
            } catch (...) {
                exec_mode = 1;
            }
        }
    }
    std::cout << "--> Wybrany tryb pracy: [" << exec_mode << "]\n" << std::endl;

    // 0b. Immersion Refractive Index
    float n_immersion = 1.52f;
    std::cout << "Enter refractive index of immersion medium (RI) [default 1.52]: ";
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
    std::cout << "--> Wybrany wspolczynnik zalamania (RI): " << n_immersion << "\n" << std::endl;

    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        // 1. Choose Object File
        std::string obj_path = (argc > 1) ? argv[1] : "";
        if (obj_path.empty() || !oct::Hdf5Loader::is_valid_hdf5(obj_path)) {
            std::cout << "[File Chooser] Select OBJECT measurement file (*.hdf5)..." << std::endl;
            obj_path = oct::FileDialog::open_file("Select OBJECT measurement file (*.hdf5)", {"hdf5", "h5"});
            if (obj_path.empty()) {
                std::cout << "Object file selection cancelled. Terminating." << std::endl;
                return 0;
            }
        }

        // 2. Choose Reference File
        std::string ref_path = (argc > 2) ? argv[2] : "";
        if (ref_path.empty() || !oct::Hdf5Loader::is_valid_hdf5(ref_path)) {
            std::cout << "[File Chooser] Select REFERENCE measurement file (*.hdf5)..." << std::endl;
            ref_path = oct::FileDialog::open_file("Select REFERENCE measurement file (*.hdf5)", {"hdf5", "h5"});
            if (ref_path.empty()) {
                std::cout << "Reference file selection cancelled. Terminating." << std::endl;
                return 0;
            }
        }

        // 3. Load Datasets into RAM
        std::cout << "[1/3] Wczytuje dane Obiektu z HDF5: " << obj_path << std::endl;
        auto ds_obj = oct::Hdf5Loader::load(obj_path);

        std::cout << "[2/3] Wczytuje dane Referencji z HDF5: " << ref_path << std::endl;
        auto ds_ref = oct::Hdf5Loader::load(ref_path);

        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // 4. Ultra-Fast Phase-OCT Reconstruction Benchmark
        std::cout << "[3/3] Start rekonstrukcji Phase-OCT..." << std::endl;

        // START STRICT BENCHMARK TIMER (Post-load memory to final phase)
        auto t_bench_start = std::chrono::high_resolution_clock::now();

        auto engine = std::make_unique<oct::Volume3DEngine>(device, ds_obj->width, ds_obj->height, ds_obj->num_wavelengths);
        oct::PhaseOctStats stats;
        
        oct::PipelineMode p_mode = (exec_mode == 3 || exec_mode == 4) ? oct::PipelineMode::Matlab_Exact_FFT : oct::PipelineMode::Optimized_DTFT;

        engine->reconstruct_phase_oct(
            ds_obj->data.data(),
            ds_ref->data.data(),
            ds_obj->wavelengths_nm,
            0.0f, n_immersion, 0.5f, false,
            p_mode,
            oct::ComputeBackend3D::CPUMultiCore,
            stats
        );

        auto t_bench_end = std::chrono::high_resolution_clock::now();
        double pure_compute_ms = std::chrono::duration<double, std::milli>(t_bench_end - t_bench_start).count();

        std::cout << "\n=================================================================" << std::endl;
        std::cout << " REKONSTRUKCJA PHASE-OCT UKONCZONA SUKCESEM!" << std::endl;
        std::cout << "  --> Czas obliczen (RAM -> Faza 2D): " << std::fixed << std::setprecision(2) 
                  << pure_compute_ms << " ms (" << std::setprecision(4) << pure_compute_ms / 1000.0 << " s)" << std::endl;
        std::cout << "  - Warstwa obiektu:     z_obj = " << stats.z_obj_index << std::endl;
        std::cout << "  - Warstwa referencji:  z_ref = " << stats.z_ref_index << std::endl;
        std::cout << "  - Zakres fazy:         [" << std::setprecision(3) << stats.phase_min_rad << " .. " << stats.phase_max_rad << "] rad" << std::endl;
        std::cout << "  - Wysokosc schodka:    " << std::setprecision(3) << stats.step_height_um << " um" << std::endl;
        std::cout << "=================================================================\n" << std::endl;

        // 5. Automatic Export to Disk (MAT + PNG z Colorbarem)
        std::string out_dir = "./output";
        std::filesystem::create_directories(out_dir);

        const auto& ph_recon = engine->get_phase_reconstructed();

        // 5a. Export MATLAB Level 5 File
        std::string mat_path = out_dir + "/reconstructed_phase.mat";
        export_mat5_matrix(mat_path, ph_recon.data(), ds_obj->width, ds_obj->height);

        // 5b. Export High-Res PNG with Colorbar (Default range [-4.0 .. +3.5] rad matching MATLAB)
        float cmin = -4.0f, cmax = 3.5f;
        const uint32_t* lut_jet = oct::ColormapManager::get_lut(oct::ColormapType::Jet);
        std::string png_path = out_dir + "/reconstructed_phase.png";
        export_png_with_colorbar(png_path, ph_recon.data(), ds_obj->width, ds_obj->height, cmin, cmax, lut_jet);

        // If Headless Mode (2 or 4), terminate immediately
        if (exec_mode == 2 || exec_mode == 4) {
            std::cout << "\n[Koniec] Wyniki zapisane w katalogu ./output. Zamykam program." << std::endl;
            return 0;
        }

        // 6. Interactive GUI Mode (1 or 4)
        if (!device) return 1;

        ReconAppDelegate* delegate = [[ReconAppDelegate alloc] init];
        [NSApp setDelegate:delegate];

        NSRect frame = NSMakeRect(60, 60, 1680, 1020);
        NSUInteger styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                               NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable;

        NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:styleMask
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        [window setTitle:@"Phase-OCT High-Speed 3D Reconstruction & Interactive Line Profile"];
        [window center];

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];
        CAMetalLayer* metalLayer = [CAMetalLayer layer];
        metalLayer.device = device;
        metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;

        ReconCustomView* contentView = [[ReconCustomView alloc] initWithFrame:frame];
        [contentView setWantsLayer:YES];
        [contentView setLayer:metalLayer];
        [window setContentView:contentView];
        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

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

        id<MTLTexture> tex_recon = create_texture_from_floats(device, ph_recon.data(), ds_obj->width, ds_obj->height, cmin, cmax, lut_jet);
        id<MTLTexture> tex_colorbar = create_colorbar_texture(device, lut_jet, 256);

        // Interactive Cross-Section Line State (normalized [0, 1])
        ImVec2 line_p0 = ImVec2(0.1f, 0.5f);
        ImVec2 line_p1 = ImVec2(0.9f, 0.5f);
        bool is_dragging_line = false;

        float yaw = 0.75f, pitch = 1.25f, zoom = 1.8f, z_scale = 3.5f;
        MTLRenderPassDescriptor* renderPassDesc = [MTLRenderPassDescriptor renderPassDescriptor];

        // 7. GUI Render Loop
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

                ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_Always);
                ImGui::SetNextWindowSize(ImVec2(bounds.size.width - 20, bounds.size.height - 20), ImGuiCond_Always);

                ImGui::Begin("Phase-OCT Production Dashboard", nullptr,
                             ImGuiWindowFlags_NoResize | ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse);

                // Header Banner
                ImGui::BeginChild("Banner", ImVec2(0, 75), true);
                ImGui::Columns(4, "BannerCols", false);

                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Compute Time (RAM -> Phase):");
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "%.2f ms (%.4f s)", pure_compute_ms, pure_compute_ms / 1000.0f);
                ImGui::Text("z_obj: %d | z_ref: %d", stats.z_obj_index, stats.z_ref_index);

                ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Optical Parameters:");
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "RI = %.3f", n_immersion);
                ImGui::Text("Background (zeroed): ~0.00 rad");

                ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Step Phase:");
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "Delta phi = %+.2f rad", stats.step_phase_delay_rad);
                ImGui::SetNextItemWidth(140);
                if (ImGui::DragFloatRange2("##Range", &cmin, &cmax, 0.1f, -20.0f, 20.0f, "Min: %.1f", "Max: %.1f")) {
                    tex_recon = create_texture_from_floats(device, ph_recon.data(), ds_obj->width, ds_obj->height, cmin, cmax, lut_jet);
                }

                ImGui::NextColumn();
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "Auto-Saved:");
                ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "output/reconstructed_phase.mat");
                ImGui::TextColored(ImVec4(0.5f, 0.9f, 0.5f, 1.0f), "output/reconstructed_phase.png");

                ImGui::Columns(1);
                ImGui::EndChild();

                // Main Views
                float avail_w = ImGui::GetContentRegionAvail().x;
                float avail_h = ImGui::GetContentRegionAvail().y;
                float col_w = (avail_w - 16.0f) * 0.5f;
                float plot_h = avail_h - 220.0f;

                // Left: 2D Phase Map + Vertical Colorbar (STRICT 1:1 Aspect Ratio)
                ImGui::BeginChild("Phase2DView", ImVec2(col_w, plot_h), true);
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "2D Phase Map (Drag mouse over image to draw a line):");

                const float cb_w_gui = 20.0f;
                const float cb_label_w = 65.0f;
                float max_avail_img_w = col_w - 32.0f - cb_w_gui - cb_label_w;
                float max_avail_img_h = plot_h - 35.0f;
                float img_square_size = std::min(max_avail_img_w, max_avail_img_h);

                ImVec2 img_pos = ImGui::GetCursorScreenPos();

                if (tex_recon) {
                    ImGui::Image((__bridge void*)tex_recon, ImVec2(img_square_size, img_square_size));

                    // Interactive Line Drawing on Mouse Drag
                    if (ImGui::IsItemHovered()) {
                        ImVec2 m = io.MousePos;
                        float norm_x = std::clamp((m.x - img_pos.x) / img_square_size, 0.0f, 1.0f);
                        float norm_y = std::clamp((m.y - img_pos.y) / img_square_size, 0.0f, 1.0f);

                        if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                            line_p0 = ImVec2(norm_x, norm_y);
                            line_p1 = ImVec2(norm_x, norm_y);
                            is_dragging_line = true;
                        }
                        if (is_dragging_line && ImGui::IsMouseDragging(ImGuiMouseButton_Left)) {
                            line_p1 = ImVec2(norm_x, norm_y);
                        }
                    }
                    if (ImGui::IsMouseReleased(ImGuiMouseButton_Left)) {
                        is_dragging_line = false;
                    }

                    // Render Line Overlay
                    ImDrawList* draw_list = ImGui::GetWindowDrawList();
                    ImVec2 s_p0 = ImVec2(img_pos.x + line_p0.x * img_square_size, img_pos.y + line_p0.y * img_square_size);
                    ImVec2 s_p1 = ImVec2(img_pos.x + line_p1.x * img_square_size, img_pos.y + line_p1.y * img_square_size);

                    draw_list->AddLine(s_p0, s_p1, IM_COL32(255, 255, 0, 255), 2.5f);
                    draw_list->AddCircleFilled(s_p0, 5.0f, IM_COL32(0, 255, 255, 255));
                    draw_list->AddCircleFilled(s_p1, 5.0f, IM_COL32(255, 0, 255, 255));
                }

                // Vertical Colorbar next to Image matching exact height
                ImGui::SameLine();
                if (tex_colorbar) {
                    ImGui::Image((__bridge void*)tex_colorbar, ImVec2(cb_w_gui, img_square_size));
                }
                ImGui::SameLine();
                ImGui::BeginGroup();
                ImGui::Text("%.2f rad", cmax);
                ImGui::Dummy(ImVec2(0, img_square_size * 0.40f));
                ImGui::Text("%.2f rad", (cmax + cmin) * 0.5f);
                ImGui::Dummy(ImVec2(0, img_square_size * 0.40f));
                ImGui::Text("%.2f rad", cmin);
                ImGui::EndGroup();

                ImGui::EndChild();

                ImGui::SameLine();

                // Right: 3D Surface Topography
                ImGui::BeginChild("Surface3DView", ImVec2(col_w, plot_h), true);
                ImGui::TextColored(ImVec4(0.35f, 0.65f, 1.0f, 1.0f), "3D Surface Topography (Drag mouse: rotate, Scroll: zoom):");
                engine->render_raymarching(yaw, pitch, zoom, 0.0f, 1.0f, z_scale, 0, lut_jet, 256);
                void* tex3d = engine->get_rendered_texture_ptr();
                if (tex3d) {
                    ImGui::Image(tex3d, ImVec2(img_square_size, img_square_size));
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
                ImGui::EndChild();

                // Bottom: Dynamic Arbitrary 1D Profile Plot along line P0 -> P1
                ImGui::BeginChild("Bottom1DPlot", ImVec2(avail_w, avail_h - plot_h - 10.0f), true);

                int x0_pix = (int)(line_p0.x * (ds_obj->width - 1));
                int y0_pix = (int)(line_p0.y * (ds_obj->height - 1));
                int x1_pix = (int)(line_p1.x * (ds_obj->width - 1));
                int y1_pix = (int)(line_p1.y * (ds_obj->height - 1));

                float dx_pix = (float)(x1_pix - x0_pix);
                float dy_pix = (float)(y1_pix - y0_pix);
                float line_len_pix = std::sqrt(dx_pix * dx_pix + dy_pix * dy_pix);
                int num_samples = std::max(2, (int)line_len_pix);

                std::vector<float> line_profile(num_samples);
                std::vector<float> line_dist_um(num_samples);
                float total_dist_um = line_len_pix * 0.5f;

                for (int s = 0; s < num_samples; ++s) {
                    float t = (num_samples > 1) ? (float)s / (float)(num_samples - 1) : 0.0f;
                    int sx = std::clamp((int)std::round(x0_pix + t * dx_pix), 0, ds_obj->width - 1);
                    int sy = std::clamp((int)std::round(y0_pix + t * dy_pix), 0, ds_obj->height - 1);
                    line_profile[s] = ph_recon[(size_t)sy * ds_obj->width + sx];
                    line_dist_um[s] = t * total_dist_um;
                }

                if (ImPlot::BeginPlot("##ArbitraryProfile", ImVec2(avail_w - 16.0f, avail_h - plot_h - 25.0f))) {
                    ImPlot::SetupAxes("Section line length [um]", "Phase [rad]");
                    ImPlot::PlotLine("Phase Profile along Drawn Line", line_dist_um.data(), line_profile.data(), num_samples);
                    ImPlot::EndPlot();
                }
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

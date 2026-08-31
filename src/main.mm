#import <Cocoa/Cocoa.h>
#import <Metal/Metal.h>
#import <MetalKit/MetalKit.h>
#include <iostream>
#include <memory>

#include "imgui.h"
#include "imgui_impl_metal.h"
#include "imgui_impl_osx.h"
#include "implot.h"

#include "config.hpp"
#include "hdf5_loader.hpp"
#include "file_dialog.hpp"
#include "dataset_loader.hpp"
#include "producer_thread.hpp"
#include "processor_thread.hpp"
#include "app_gui.hpp"

@interface AppDelegate : NSObject <NSApplicationDelegate>
@end

@implementation AppDelegate
- (BOOL)applicationShouldTerminateAfterLastWindowClosed:(NSApplication *)sender {
    return YES;
}
@end

@interface CustomView : NSView
@end

@implementation CustomView
- (BOOL)acceptsFirstResponder { return YES; }
@end

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        [NSApplication sharedApplication];
        [NSApp setActivationPolicy:NSApplicationActivationPolicyRegular];

        AppDelegate* delegate = [[AppDelegate alloc] init];
        [NSApp setDelegate:delegate];

        // 1. Create Native macOS Window
        NSRect frame = NSMakeRect(100, 100, 1600, 1000);
        NSUInteger styleMask = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable |
                               NSWindowStyleMaskResizable | NSWindowStyleMaskMiniaturizable;

        NSWindow* window = [[NSWindow alloc] initWithContentRect:frame
                                                       styleMask:styleMask
                                                         backing:NSBackingStoreBuffered
                                                           defer:NO];
        [window setTitle:@"Swept-Source Full-Field OCT (SS-FF-OCT) Real-time Simulator [Native C++ / Metal]"];
        [window center];

        // 2. Setup Metal Device & CAMetalLayer
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::cerr << "Failed to create Metal device." << std::endl;
            return 1;
        }

        id<MTLCommandQueue> commandQueue = [device newCommandQueue];

        CAMetalLayer* metalLayer = [CAMetalLayer layer];
        metalLayer.device = device;
        metalLayer.pixelFormat = MTLPixelFormatBGRA8Unorm;

        CustomView* contentView = [[CustomView alloc] initWithFrame:frame];
        [contentView setWantsLayer:YES];
        [contentView setLayer:metalLayer];
        [window setContentView:contentView];
        [window makeKeyAndOrderFront:nil];
        [NSApp activateIgnoringOtherApps:YES];

        // 3. Initialize Dear ImGui & ImPlot
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImPlot::CreateContext();

        ImGuiIO& io = ImGui::GetIO();
        io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

        // Apply sleek dark theme
        ImGui::StyleColorsDark();
        ImGuiStyle& style = ImGui::GetStyle();
        style.WindowRounding = 6.0f;
        style.FrameRounding = 4.0f;
        style.GrabRounding = 4.0f;
        style.ItemSpacing = ImVec2(8, 6);
        style.Colors[ImGuiCol_WindowBg] = ImVec4(0.06f, 0.08f, 0.11f, 1.00f);
        style.Colors[ImGuiCol_ChildBg] = ImVec4(0.09f, 0.11f, 0.15f, 1.00f);
        style.Colors[ImGuiCol_Header] = ImVec4(0.12f, 0.35f, 0.65f, 0.80f);
        style.Colors[ImGuiCol_Button] = ImVec4(0.15f, 0.20f, 0.27f, 1.00f);
        style.Colors[ImGuiCol_ButtonHovered] = ImVec4(0.20f, 0.35f, 0.55f, 1.00f);
        style.Colors[ImGuiCol_ButtonActive] = ImVec4(0.12f, 0.45f, 0.85f, 1.00f);
        style.Colors[ImGuiCol_SliderGrab] = ImVec4(0.20f, 0.55f, 0.95f, 1.00f);
        style.Colors[ImGuiCol_SliderGrabActive] = ImVec4(0.35f, 0.70f, 1.00f, 1.00f);

        ImGui_ImplOSX_Init(contentView);
        ImGui_ImplMetal_Init(device);

        // 3a. Prompt for PipelineMode
        int exec_mode = 1;
        std::cout << "Select Pipeline Mode:\n"
                  << "  [1] Optimized C++ with GUI (High speed + Miguel 2D)\n"
                  << "  [3] MATLAB Exact with GUI (Maximum precision, slower)\n"
                  << "Select mode [1, 3, default 1]: ";
        std::cout.flush();

        std::string mode_input;
        if (std::getline(std::cin, mode_input)) {
            mode_input.erase(std::remove_if(mode_input.begin(), mode_input.end(), ::isspace), mode_input.end());
            if (!mode_input.empty()) {
                try {
                    exec_mode = std::stoi(mode_input);
                    if (exec_mode != 1 && exec_mode != 3) exec_mode = 1;
                } catch (...) {
                    exec_mode = 1;
                }
            }
        }
        std::cout << "--> Wybrany tryb potoku: [" << exec_mode << "]\n" << std::endl;
        oct::PipelineMode p_mode = (exec_mode == 3) ? oct::PipelineMode::Matlab_Exact_FFT : oct::PipelineMode::Optimized_DTFT;

        // 4. Prompt user to select HDF5 measurement file
        std::string dataset_path = (argc > 1) ? argv[1] : "";
        if (dataset_path.empty() || !oct::Hdf5Loader::is_valid_hdf5(dataset_path)) {
            std::cout << "[Main] Opening native OS File Selection Dialog..." << std::endl;
            dataset_path = oct::FileDialog::open_file("Select OCT Measurement File (*.hdf5)");
            if (dataset_path.empty()) {
                std::cout << "[Main] No file selected. Exiting." << std::endl;
                return 0;
            }
        }

        std::cout << "[Main] Loading full 1504x1504 SS-OCT volume from HDF5: " << dataset_path << std::endl;
        auto dataset = oct::Hdf5Loader::load(dataset_path);

        auto raw_frame_queue = std::make_shared<oct::ThreadSafeQueue<oct::RawFrameMessage>>(oct::RAW_FRAME_QUEUE_MAXSIZE);
        auto results_queue = std::make_shared<oct::ThreadSafeQueue<oct::ProcessedResult>>(oct::RESULTS_QUEUE_MAXSIZE);

        auto producer = std::make_shared<oct::ProducerThread>(dataset, raw_frame_queue);
        auto processor = std::make_shared<oct::ProcessorThread>(
            raw_frame_queue, results_queue, dataset->num_wavelengths, dataset->height, dataset->width
        );
        processor->set_wavelengths(dataset->wavelengths_nm);

        auto app_gui = std::make_unique<oct::AppGUI>(device, producer, processor, results_queue, p_mode);

        // Start isolated threads
        producer->start();
        processor->start();

        MTLRenderPassDescriptor* renderPassDescriptor = [MTLRenderPassDescriptor renderPassDescriptor];

        // 5. Metal 120 FPS Render Loop
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

                // Window dimensions
                NSRect bounds = [contentView bounds];
                if (bounds.size.width <= 0 || bounds.size.height <= 0) {
                    [NSThread sleepForTimeInterval:0.01];
                    continue;
                }

                metalLayer.drawableSize = CGSizeMake(bounds.size.width * [window backingScaleFactor],
                                                     bounds.size.height * [window backingScaleFactor]);

                id<CAMetalDrawable> drawable = [metalLayer nextDrawable];
                if (!drawable) continue;

                renderPassDescriptor.colorAttachments[0].texture = drawable.texture;
                renderPassDescriptor.colorAttachments[0].loadAction = MTLLoadActionClear;
                renderPassDescriptor.colorAttachments[0].clearColor = MTLClearColorMake(0.04, 0.05, 0.07, 1.0);
                renderPassDescriptor.colorAttachments[0].storeAction = MTLStoreActionStore;

                id<MTLCommandBuffer> commandBuffer = [commandQueue commandBuffer];
                id<MTLRenderCommandEncoder> renderEncoder = [commandBuffer renderCommandEncoderWithDescriptor:renderPassDescriptor];

                ImGui_ImplMetal_NewFrame(renderPassDescriptor);
                ImGui_ImplOSX_NewFrame(contentView);
                ImGui::NewFrame();

                // Render GUI & Viewports
                app_gui->update_and_render();

                ImGui::Render();
                ImGui_ImplMetal_RenderDrawData(ImGui::GetDrawData(), commandBuffer, renderEncoder);

                [renderEncoder endEncoding];
                [commandBuffer presentDrawable:drawable];
                [commandBuffer commit];
            }
        }

        // Cleanup on exit
        std::cout << "[Main] Stopping simulation threads..." << std::endl;
        producer->stop();
        processor->stop();

        app_gui.reset();

        ImGui_ImplMetal_Shutdown();
        ImGui_ImplOSX_Shutdown();
        ImPlot::DestroyContext();
        ImGui::DestroyContext();
    }
    return 0;
}

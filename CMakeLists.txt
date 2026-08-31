cmake_minimum_required(VERSION 3.22)
project(SS_FF_OCT_Simulator VERSION 2.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_CUDA "Enable NVIDIA CUDA backend (Windows / Linux)" ON)

# Find HDF5
find_package(HDF5 REQUIRED COMPONENTS C)
include_directories(${HDF5_INCLUDE_DIRS})

# Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/vendor/implot
    /opt/homebrew/include
)
link_directories(/opt/homebrew/lib)

# Common Sources
set(COMMON_SOURCES
    src/hdf5_loader.cpp
    src/dataset_loader.cpp
    src/colormaps.cpp
    src/producer_thread.cpp
    src/cpu_compute_engine.cpp
    src/processor_thread.cpp
    vendor/imgui/imgui.cpp
    vendor/imgui/imgui_draw.cpp
    vendor/imgui/imgui_tables.cpp
    vendor/imgui/imgui_widgets.cpp
    vendor/implot/implot.cpp
    vendor/implot/implot_items.cpp
)

# -------------------------------------------------------------
# macOS Target (Apple Silicon Metal + Accelerate + Cocoa)
# -------------------------------------------------------------
if(APPLE)
    enable_language(OBJCXX)
    set(MACOS_SOURCES
        src/metal_compute_engine.mm
        src/volume3d_engine.mm
        src/file_dialog.mm
        src/app_gui.mm
        src/main.mm
        vendor/imgui/imgui_impl_osx.mm
        vendor/imgui/imgui_impl_metal.mm
    )

    add_executable(oct_simulator ${COMMON_SOURCES} ${MACOS_SOURCES})
    target_compile_options(oct_simulator PRIVATE -O3 -Wall -Wno-deprecated-declarations -DACCELERATE_NEW_LAPACK -DIMGUI_IMPL_OSX_DISABLE_GAMEPAD -fobjc-arc)
    target_link_libraries(oct_simulator PRIVATE
        ${HDF5_LIBRARIES}
        zstd
        "-framework Cocoa"
        "-framework Metal"
        "-framework MetalKit"
        "-framework Accelerate"
        "-framework QuartzCore"
        "-framework GameController"
    )

# -------------------------------------------------------------
# Windows Target (MSVC / Clang + NVIDIA CUDA + Direct3D11 / OpenGL)
# -------------------------------------------------------------
elseif(WIN32)
    if(ENABLE_CUDA)
        include(CheckLanguage)
        check_language(CUDA)
        if(CMAKE_CUDA_COMPILER)
            enable_language(CUDA)
            set(CMAKE_CUDA_STANDARD 20)
            set(CMAKE_CUDA_ARCHITECTURES "89;86;80;75") # RTX 4090 (Ada Lovelace sm_89), RTX 3000 (sm_86), etc.
            list(APPEND COMMON_SOURCES src/cuda_compute_engine.cu)
            add_definitions(-DOCT_ENABLE_CUDA)
        endif()
    endif()

    set(WINDOWS_SOURCES
        src/file_dialog_win.cpp
        src/main_win.cpp
        vendor/imgui/imgui_impl_win32.cpp
        vendor/imgui/imgui_impl_dx11.cpp
    )

    add_executable(oct_simulator ${COMMON_SOURCES} ${WINDOWS_SOURCES})
    target_compile_options(oct_simulator PRIVATE /O2 /W3 /openmp)
    target_link_libraries(oct_simulator PRIVATE
        ${HDF5_LIBRARIES}
        d3d11
        dxgi
        comdlg32
    )
    if(CMAKE_CUDA_COMPILER)
        target_link_libraries(oct_simulator PRIVATE CUDA::cufft CUDA::cudart)
    endif()

    add_library(phase_oct SHARED ${COMMON_SOURCES} src/volume3d_engine_win.cpp src/python_api.cpp src/miguel_unwrapper.cpp)
    target_compile_options(phase_oct PRIVATE /O2 /W3 /openmp)
    target_link_libraries(phase_oct PRIVATE ${HDF5_LIBRARIES})
    if(CMAKE_CUDA_COMPILER)
        target_link_libraries(phase_oct PRIVATE CUDA::cufft CUDA::cudart)
    endif()
    set_target_properties(phase_oct PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin" LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")
endif()

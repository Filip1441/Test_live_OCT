# Phase-OCT Reconstruction Engine

A high-speed cross-platform engine for Phase-OCT processing and 3D surface topography reconstruction. 
The core is written in C++ (with Apple Metal and CPU multithreading support) and features a seamless Python API for data integration (FAIR principles).

## Requirements

### Python Environment
- Python 3.12+ (tested with 3.14 on macOS)
- `numpy`, `matplotlib`, `fair_core`

### C++ Build System
- **macOS**: Clang, CMake / Make, Apple Accelerate, Apple Metal
- **Windows**: MSVC / Clang, CMake, (Optional: NVIDIA CUDA)
- **Dependencies**: HDF5, ZSTD (typically installed via Homebrew or vcpkg)

## Installation & Build

### macOS (Using Make)
```bash
make recon
make run # to launch simulator GUI
```
Or to build the shared library required for Python:
```bash
make bin/libphase_oct.dylib
```

### Windows (Using CMake)
```bash
mkdir build
cd build
cmake ..
cmake --build . --config Release
```

## Running Python Integration

Make sure you have activated your virtual environment:
```bash
source .venv/bin/activate
```

Run the reconstruction engine via Python:
```bash
python OCT_recon.py --mode matlab
```
Options:
- `--mode [matlab | fast]`: Select the reconstruction pipeline.
- `--gui`: Display the reconstructed phase map via matplotlib.
- `--dont_save`: Skip saving the metadata and results.

## Project Structure
- `OCT_recon.py`: Main python pipeline script.
- `src/`: C++ source code for compute engine and data handling.
- `include/`: C++ headers.
- `tests/` & `tools/`: Unit tests and diagnostics.

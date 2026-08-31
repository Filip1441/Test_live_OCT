#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include "miguel_unwrapper.hpp"

// Simple tool to read a flat array of 32-bit floats, run Miguel unwrapper, and save it.
int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <input.bin> <width> <height> <output.bin>\n";
        return 1;
    }
    
    std::string in_file = argv[1];
    int width = std::stoi(argv[2]);
    int height = std::stoi(argv[3]);
    std::string out_file = argv[4];
    
    size_t total = (size_t)width * height;
    std::vector<float> data(total);
    
    std::ifstream is(in_file, std::ios::binary);
    if (!is.read((char*)data.data(), total * sizeof(float))) {
        std::cerr << "Failed to read input\n";
        return 1;
    }
    is.close();
    
    std::cout << "Running C++ Miguel unwrapper on " << width << "x" << height << " data...\n";
    oct::MiguelUnwrapper2D::unwrap(data, width, height);
    
    std::ofstream os(out_file, std::ios::binary);
    if (!os.write((char*)data.data(), total * sizeof(float))) {
        std::cerr << "Failed to write output\n";
        return 1;
    }
    os.close();
    
    std::cout << "Done!\n";
    return 0;
}

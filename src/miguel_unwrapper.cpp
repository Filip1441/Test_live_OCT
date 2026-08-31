#include "miguel_unwrapper.hpp"
#include <cmath>
#include <algorithm>
#include <vector>
#include <chrono>
#include <iostream>
#include <cstdint>
#include <cstring>

namespace oct {

namespace {
    static float PI = 3.141592654f;
    static float TWOPI = 6.283185307f;

    struct EDGE {
        float reliab;
        int pointer_1;
        int pointer_2;
        int increment;
    };

    struct DisjointSet {
        std::vector<int> parent;
        std::vector<int> offset;
        std::vector<int> rank;

        DisjointSet(int n) : parent(n), offset(n, 0), rank(n, 0) {
            for (int i = 0; i < n; i++) parent[i] = i;
        }

        int find(int i, int& out_offset) {
            if (parent[i] == i) {
                out_offset = 0;
                return i;
            }
            int parent_offset = 0;
            int root = find(parent[i], parent_offset);
            offset[i] += parent_offset;
            parent[i] = root;
            out_offset = offset[i];
            return root;
        }

        void unite(int u, int v, int increment_uv) {
            int offset_u, offset_v;
            int root_u = find(u, offset_u);
            int root_v = find(v, offset_v);

            if (root_u != root_v) {
                if (rank[root_u] < rank[root_v]) {
                    parent[root_u] = root_v;
                    offset[root_u] = offset_v - offset_u + increment_uv;
                } else if (rank[root_u] > rank[root_v]) {
                    parent[root_v] = root_u;
                    offset[root_v] = offset_u - offset_v - increment_uv;
                } else {
                    parent[root_u] = root_v;
                    offset[root_u] = offset_v - offset_u + increment_uv;
                    rank[root_v]++;
                }
            }
        }
    };

    int find_wrap(float pixel1, float pixel2) {
        float difference = pixel1 - pixel2;
        if (difference > PI) return -1;
        else if (difference < -PI) return 1;
        return 0;
    }

    float wrap(float pixel_value) {
        float wrapped_pixel_value = pixel_value;
        if (pixel_value > PI) wrapped_pixel_value -= TWOPI;
        if (pixel_value < -PI) wrapped_pixel_value += TWOPI;
        return wrapped_pixel_value;
    }

    float frac(float pixel_value) {
        float frac_value = pixel_value;
        if (pixel_value > PI) frac_value -= TWOPI;
        if (pixel_value < -PI) frac_value += TWOPI;
        return frac_value;
    }

    float second_difference(float pixel1, float pixel2, float pixel3) {
        float second_diff = wrap(pixel1 - pixel2) - wrap(pixel2 - pixel3);
        return frac(second_diff);
    }

    void calculate_reliability(const std::vector<float>& wrappedImage, std::vector<float>& reliability, int image_width, int image_height) {
        int image_width_plus_one = image_width + 1;
        int image_width_minus_one = image_width - 1;

        for (int i = 1; i < image_height - 1; ++i) {
            for (int j = 1; j < image_width - 1; ++j) {
                int idx = i * image_width + j;
                
                float H = second_difference(wrappedImage[idx - 1], wrappedImage[idx], wrappedImage[idx + 1]);
                float V = second_difference(wrappedImage[idx - image_width], wrappedImage[idx], wrappedImage[idx + image_width]);
                float D1 = second_difference(wrappedImage[idx - image_width_plus_one], wrappedImage[idx], wrappedImage[idx + image_width_plus_one]);
                float D2 = second_difference(wrappedImage[idx - image_width_minus_one], wrappedImage[idx], wrappedImage[idx + image_width_minus_one]);

                reliability[idx] = H * H + V * V + D1 * D1 + D2 * D2;
            }
        }
    }
    
    void radix_sort_edges(std::vector<EDGE>& edge, int no_edges) {
        std::vector<EDGE> temp(no_edges);
        EDGE* in = edge.data();
        EDGE* out = temp.data();

        int count[2048];
        
        // Pass 1: bits 0-10
        std::memset(count, 0, sizeof(count));
        for (int i = 0; i < no_edges; i++) {
            uint32_t val;
            std::memcpy(&val, &in[i].reliab, sizeof(float));
            count[val & 0x7FF]++;
        }
        for (int i = 1; i < 2048; i++) count[i] += count[i - 1];
        for (int i = no_edges - 1; i >= 0; i--) {
            uint32_t val;
            std::memcpy(&val, &in[i].reliab, sizeof(float));
            out[--count[val & 0x7FF]] = in[i];
        }

        // Pass 2: bits 11-21
        std::memset(count, 0, sizeof(count));
        for (int i = 0; i < no_edges; i++) {
            uint32_t val;
            std::memcpy(&val, &out[i].reliab, sizeof(float));
            count[(val >> 11) & 0x7FF]++;
        }
        for (int i = 1; i < 2048; i++) count[i] += count[i - 1];
        for (int i = no_edges - 1; i >= 0; i--) {
            uint32_t val;
            std::memcpy(&val, &out[i].reliab, sizeof(float));
            in[--count[(val >> 11) & 0x7FF]] = out[i];
        }

        // Pass 3: bits 22-31
        std::memset(count, 0, sizeof(count));
        for (int i = 0; i < no_edges; i++) {
            uint32_t val;
            std::memcpy(&val, &in[i].reliab, sizeof(float));
            count[(val >> 22) & 0x3FF]++;
        }
        for (int i = 1; i < 1024; i++) count[i] += count[i - 1];
        for (int i = no_edges - 1; i >= 0; i--) {
            uint32_t val;
            std::memcpy(&val, &in[i].reliab, sizeof(float));
            out[--count[(val >> 22) & 0x3FF]] = in[i];
        }
        
        std::memcpy(in, out, no_edges * sizeof(EDGE));
    }
} // anonymous namespace

void MiguelUnwrapper2D::unwrap(std::vector<float>& phase, int width, int height) {
    if (width <= 1 || height <= 1) return;

    auto t0 = std::chrono::high_resolution_clock::now();

    int image_size = height * width;
    int No_of_Edges = (width) * (height - 1) + (width - 1) * (height);

    std::vector<float> reliability(image_size, 9999999.0f);
    std::vector<EDGE> edge(No_of_Edges);

    auto t1 = std::chrono::high_resolution_clock::now();
    calculate_reliability(phase, reliability, width, height);
    
    int edge_idx = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width - 1; j++) {
            int idx = i * width + j;
            edge[edge_idx].pointer_1 = idx;
            edge[edge_idx].pointer_2 = idx + 1;
            edge[edge_idx].reliab = reliability[idx] + reliability[idx + 1];
            edge[edge_idx].increment = find_wrap(phase[idx], phase[idx + 1]);
            edge_idx++;
        }
    }
    for (int i = 0; i < height - 1; i++) {
        for (int j = 0; j < width; j++) {
            int idx = i * width + j;
            edge[edge_idx].pointer_1 = idx;
            edge[edge_idx].pointer_2 = idx + width;
            edge[edge_idx].reliab = reliability[idx] + reliability[idx + width];
            edge[edge_idx].increment = find_wrap(phase[idx], phase[idx + width]);
            edge_idx++;
        }
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    radix_sort_edges(edge, No_of_Edges);
    
    auto t3 = std::chrono::high_resolution_clock::now();
    DisjointSet ds(image_size);
    for (int i = 0; i < No_of_Edges; i++) {
        ds.unite(edge[i].pointer_1, edge[i].pointer_2, edge[i].increment);
    }
    
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < image_size; i++) {
        int dummy;
        ds.find(i, dummy);
        phase[i] += TWOPI * (float)(ds.offset[i]);
    }
    
    auto t5 = std::chrono::high_resolution_clock::now();
    
    std::cout << "[Unwrap Profiling] Alloc & Reliab: " << std::chrono::duration<double, std::milli>(t1-t0).count() << " ms\n";
    std::cout << "[Unwrap Profiling] Create Edges: " << std::chrono::duration<double, std::milli>(t2-t1).count() << " ms\n";
    std::cout << "[Unwrap Profiling] Radix Sort: " << std::chrono::duration<double, std::milli>(t3-t2).count() << " ms\n";
    std::cout << "[Unwrap Profiling] UF Gather: " << std::chrono::duration<double, std::milli>(t4-t3).count() << " ms\n";
    std::cout << "[Unwrap Profiling] Apply: " << std::chrono::duration<double, std::milli>(t5-t4).count() << " ms\n";
}

void MiguelUnwrapper2D::detrend_median_gradient(std::vector<float>& phase, int width, int height) {
    if (width <= 1 || height <= 1) return;
    
    std::vector<float> dx; dx.reserve((size_t)(width - 1) * height);
    std::vector<float> dy; dy.reserve((size_t)width * (height - 1));

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width - 1; ++x) {
            dx.push_back(phase[(size_t)y * width + x + 1] - phase[(size_t)y * width + x]);
        }
    }
    for (int y = 0; y < height - 1; ++y) {
        for (int x = 0; x < width; ++x) {
            dy.push_back(phase[(size_t)(y + 1) * width + x] - phase[(size_t)y * width + x]);
        }
    }

    auto get_median = [](std::vector<float>& vec) {
        size_t n = vec.size() / 2;
        std::nth_element(vec.begin(), vec.begin() + n, vec.end());
        return vec[n];
    };

    float med_dx = get_median(dx);
    float med_dy = get_median(dy);

    float center_x = (width - 1) / 2.0f;
    float center_y = (height - 1) / 2.0f;

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float tilt = med_dx * (x - center_x) + med_dy * (y - center_y);
            phase[(size_t)y * width + x] -= tilt;
        }
    }
}

} // namespace oct

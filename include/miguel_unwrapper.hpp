#pragma once

#include <vector>

namespace oct {

class MiguelUnwrapper2D {
public:
    // Unwraps the phase in-place using Reliability-Guided MST (Herráez/Miguel approach).
    // The input phase array must be of size width * height.
    static void unwrap(std::vector<float>& phase, int width, int height);

    // Removes phase ramp/tilt based on median gradient (Detrend_medGradient equivalent).
    static void detrend_median_gradient(std::vector<float>& phase, int width, int height);

private:
    struct Edge {
        int u;
        int v;
        float weight;
        bool operator<(const Edge& other) const {
            return weight > other.weight; // Higher reliability (weight) first, or lower weight if we use second diffs.
            // We will use lower weight = higher reliability.
        }
    };
};

} // namespace oct

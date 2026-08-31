#pragma once

#include <string>
#include <cstdint>

namespace oct {

// Paths
inline const std::string DEFAULT_CONFIG_PATH = "/Users/filip/Library/Application Support/LocalViewer/config.json";
inline const std::string DEFAULT_DATASET_UUID = "95d9e71e-22e7-4e1d-947b-94f33b65f3a4";
inline constexpr const char* DEFAULT_DATASET_PATH = "/Users/filip/Downloads/schodki_5um_ROI1504_803-878nm_751wavl_1ang_Exp180___2026-8-25_16-50-3.hdf5";

// Volume & Sensor Dimensions (Full uncropped resolution)
constexpr int RAW_WIDTH = 1504;
constexpr int RAW_HEIGHT = 1504;
constexpr int SWEEP_POINTS = 751;
constexpr int DEPTH_POINTS = 376; // (751 / 2 + 1)

// Simulation Defaults
constexpr float DEFAULT_TUNING_SPEED_NM_S = 20.0f;
constexpr float MIN_TUNING_SPEED_NM_S = 1.0f;
constexpr float MAX_TUNING_SPEED_NM_S = 500.0f;
constexpr float DEFAULT_LAMBDA_STEP_NM = 0.1f;

constexpr bool DEFAULT_DRIFT_ENABLED = false;
constexpr float DRIFT_MAX_OFFSET_PX = 40.0f;

constexpr bool DEFAULT_DROP_SIMULATION_ENABLED = false;
constexpr float DROP_PROBABILITY = 0.03f;
constexpr float CORRUPT_PROBABILITY = 0.02f;
constexpr float DUPLICATE_PROBABILITY = 0.02f;

// Visualization Defaults
constexpr float DEFAULT_DB_MIN = 20.0f;
constexpr float DEFAULT_DB_MAX = 80.0f;

// Queue Capacities
constexpr size_t RAW_FRAME_QUEUE_MAXSIZE = 2000;
constexpr size_t RESULTS_QUEUE_MAXSIZE = 3;

} // namespace oct

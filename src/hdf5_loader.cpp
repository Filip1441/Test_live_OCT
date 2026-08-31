#include "hdf5_loader.hpp"
#include <hdf5.h>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <algorithm>
#include <chrono>

namespace oct {

static float read_attr_float(hid_t grp, const char* name, float default_val) {
    if (H5Aexists(grp, name) <= 0) return default_val;
    hid_t attr = H5Aopen(grp, name, H5P_DEFAULT);
    if (attr < 0) return default_val;

    float val = default_val;
    H5Aread(attr, H5T_NATIVE_FLOAT, &val);
    H5Aclose(attr);
    return val;
}

bool Hdf5Loader::is_valid_hdf5(const std::string& path) {
    htri_t is_h5 = H5Fis_hdf5(path.c_str());
    return (is_h5 > 0);
}

std::shared_ptr<Dataset> Hdf5Loader::load(
    const std::string& hdf5_path,
    bool is_reference,
    int max_frames
) {
    auto t0 = std::chrono::high_resolution_clock::now();
    H5Eset_auto(H5E_DEFAULT, NULL, NULL);
    std::cout << "[Hdf5Loader] Opening HDF5 measurement: " << hdf5_path << std::endl;

    if (!is_valid_hdf5(hdf5_path)) {
        throw std::runtime_error("Invalid or non-existent HDF5 file: " + hdf5_path);
    }

    hid_t file_id = H5Fopen(hdf5_path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        throw std::runtime_error("Failed to open HDF5 file: " + hdf5_path);
    }

    std::string group_name = is_reference ? "/reference" : "/object";
    if (H5Lexists(file_id, group_name.c_str(), H5P_DEFAULT) <= 0) {
        if (H5Lexists(file_id, "/object", H5P_DEFAULT) > 0) {
            group_name = "/object";
        } else if (H5Lexists(file_id, "/reference", H5P_DEFAULT) > 0) {
            group_name = "/reference";
        } else {
            group_name = "";
        }
    }

    hid_t grp_id = group_name.empty() ? file_id : H5Gopen2(file_id, group_name.c_str(), H5P_DEFAULT);
    if (grp_id < 0) grp_id = file_id;

    // Read optical and dimensions attributes
    float spec_samples_f = read_attr_float(grp_id, "spectrumSamples", 751.0f);
    float sweep_start_wl = read_attr_float(grp_id, "sweepStartWavL", 878.0f);
    float sweep_stop_wl = read_attr_float(grp_id, "sweepStopWavL", 803.0f);
    float n_immersion = read_attr_float(grp_id, "immersionRefrIndex", 1.33f);
    float na_val = read_attr_float(grp_id, "NA", 1.3f);

    int target_lambdas = (int)spec_samples_f;
    if (target_lambdas <= 0) target_lambdas = 751;
    if (max_frames > 0 && target_lambdas > max_frames) {
        target_lambdas = max_frames;
    }

    // Open dataset: try group_name + "/images", then "/images", then "/data"
    std::string ds_name = group_name.empty() ? "/images" : (group_name + "/images");
    if (H5Lexists(file_id, ds_name.c_str(), H5P_DEFAULT) <= 0) {
        if (H5Lexists(file_id, "/object/images", H5P_DEFAULT) > 0) ds_name = "/object/images";
        else if (H5Lexists(file_id, "/reference/images", H5P_DEFAULT) > 0) ds_name = "/reference/images";
        else if (H5Lexists(file_id, "/images", H5P_DEFAULT) > 0) ds_name = "/images";
    }

    hid_t ds_id = H5Dopen2(file_id, ds_name.c_str(), H5P_DEFAULT);
    if (ds_id < 0) {
        if (grp_id != file_id) H5Gclose(grp_id);
        H5Fclose(file_id);
        throw std::runtime_error("Failed to open dataset '" + ds_name + "' in " + hdf5_path);
    }

    hid_t filespace = H5Dget_space(ds_id);
    int rank = H5Sget_simple_extent_ndims(filespace);
    if (rank != 3) {
        H5Sclose(filespace);
        H5Dclose(ds_id);
        H5Gclose(grp_id);
        H5Fclose(file_id);
        throw std::runtime_error("Expected 3D dataset in " + ds_name);
    }

    hsize_t dims[3];
    H5Sget_simple_extent_dims(filespace, dims, NULL);
    int total_raw_frames = (int)dims[0];
    int height = (int)dims[1];
    int width = (int)dims[2];

    // Intelligently determine 1st forward sweep: always take half if duplicated
    int frames_to_load = total_raw_frames;
    if (total_raw_frames > 1 && total_raw_frames % 2 == 0) {
        frames_to_load = total_raw_frames / 2;
    } else if (target_lambdas > 0 && target_lambdas < total_raw_frames) {
        frames_to_load = target_lambdas;
    }

    std::cout << "[Hdf5Loader] Native dimensions: " << total_raw_frames << " x " << height << " x " << width 
              << " -> Intelligently loaded 1st sweep: " << frames_to_load << " frames." << std::endl;

    // Slicing first half (1st sweep): [0..frames_to_load-1, :, :]
    hsize_t offset[3] = {0, 0, 0};
    hsize_t count[3] = {(hsize_t)frames_to_load, (hsize_t)height, (hsize_t)width};
    H5Sselect_hyperslab(filespace, H5S_SELECT_SET, offset, NULL, count, NULL);

    // Memory dataspace
    hid_t memspace = H5Screate_simple(3, count, NULL);

    auto dataset = std::make_shared<Dataset>();
    dataset->num_wavelengths = frames_to_load;
    dataset->height = height;
    dataset->width = width;
    dataset->metadata.refractive_index = n_immersion;
    dataset->metadata.na = na_val;
    dataset->metadata.raw_dtype = "float32";

    size_t total_elements = (size_t)frames_to_load * height * width;
    dataset->data.resize(total_elements);

    // Read directly into float32 (HDF5 performs automatic uint16 -> float32 conversion)
    herr_t status = H5Dread(ds_id, H5T_NATIVE_FLOAT, memspace, filespace, H5P_DEFAULT, dataset->data.data());

    H5Sclose(memspace);
    H5Sclose(filespace);
    H5Dclose(ds_id);

    if (status < 0) {
        if (grp_id != file_id) H5Gclose(grp_id);
        H5Fclose(file_id);
        throw std::runtime_error("H5Dread failed on " + ds_name);
    }

    // Check for explicit wavelengths vector in file (e.g. /vectors/wavelengths_nm, /wavelengths_nm, etc.)
    bool found_explicit_wl = false;
    std::string wl_ds_names[] = {"/vectors/wavelengths_nm", "/vectors/wavelengths", "/wavelengths_nm", "/object/wavelengths_nm"};
    for (const auto& wlname : wl_ds_names) {
        if (H5Lexists(file_id, wlname.c_str(), H5P_DEFAULT) > 0) {
            hid_t wlds = H5Dopen2(file_id, wlname.c_str(), H5P_DEFAULT);
            if (wlds >= 0) {
                hid_t wspace = H5Dget_space(wlds);
                hsize_t wdims[1];
                H5Sget_simple_extent_dims(wspace, wdims, NULL);
                int total_wl_count = (int)wdims[0];

                std::vector<double> full_wl(total_wl_count);
                if (H5Dread(wlds, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, full_wl.data()) >= 0) {
                    dataset->wavelengths_nm.resize(frames_to_load);
                    for (int i = 0; i < frames_to_load && i < total_wl_count; ++i) {
                        dataset->wavelengths_nm[i] = full_wl[i];
                    }
                    found_explicit_wl = true;
                    std::cout << "[Hdf5Loader] Extracted explicit wavelength vector (first half: " 
                              << dataset->wavelengths_nm.front() << " nm .. " << dataset->wavelengths_nm.back() << " nm)" << std::endl;
                }
                H5Sclose(wspace);
                H5Dclose(wlds);
                if (found_explicit_wl) break;
            }
        }
    }

    if (grp_id != file_id) H5Gclose(grp_id);
    H5Fclose(file_id);

    // If no explicit vector found, compute linear interpolation
    if (!found_explicit_wl) {
        float wl_min = std::min(sweep_start_wl, sweep_stop_wl);
        float wl_max = std::max(sweep_start_wl, sweep_stop_wl);
        if (std::abs(wl_max - wl_min) < 1e-3f) {
            wl_min = 803.0f;
            wl_max = 878.0f;
        }

        dataset->wavelengths_nm.resize(frames_to_load);
        for (int i = 0; i < frames_to_load; ++i) {
            float frac = (frames_to_load > 1) ? (float)i / (float)(frames_to_load - 1) : 0.0f;
            dataset->wavelengths_nm[i] = wl_min + frac * (wl_max - wl_min);
        }
    }

    if (dataset->wavelengths_nm.size() > 1) {
        dataset->lambda_step_nm = (float)std::abs(dataset->wavelengths_nm[1] - dataset->wavelengths_nm[0]);
    } else {
        dataset->lambda_step_nm = 0.1f;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double load_sec = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "[Hdf5Loader] Successfully loaded native HDF5 dataset in " << load_sec << " s ("
              << dataset->wavelengths_nm.front() << " nm .. " << dataset->wavelengths_nm.back() << " nm, step: "
              << dataset->lambda_step_nm << " nm)" << std::endl;

    return dataset;
}

} // namespace oct

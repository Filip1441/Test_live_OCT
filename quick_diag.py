#!/usr/bin/env python3
"""Quick raw phase diagnostic — no auto-alignment, just plot raw C++ output."""
import subprocess
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

obj = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.hdf5"
ref = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/reference___2026-8-26_16-48-32.hdf5"
mat = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/phase_schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.mat"

# Run mode 2 (DTFT) and mode 4 (MATLAB exact)
for mode, name in [(2, "DTFT"), (4, "MATLAB_Exact")]:
    proc = subprocess.Popen(
        ['./bin/phase_oct_recon', obj, ref],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    proc.communicate(f"{mode}\n1.52\n")
    
    cpp_path = "output/reconstructed_phase.mat"
    cpp = sio.loadmat(cpp_path)['ph_reconstructed']
    
    print(f"Mode {mode} ({name}): shape={cpp.shape}, min={cpp.min():.2f}, max={cpp.max():.2f}, mean={cpp.mean():.2f}")
    
    # Save for comparison
    np.save(f"raw_mode{mode}.npy", cpp)

# Load MATLAB target
with h5py.File(mat, 'r') as f:
    ph_mat = np.array(f['ph_initial']).T
print(f"MATLAB ph_initial: shape={ph_mat.shape}, min={ph_mat.min():.2f}, max={ph_mat.max():.2f}, mean={ph_mat.mean():.2f}")

# Plot all 3 side by side - RAW, no auto-alignment
m2 = np.load("raw_mode2.npy")
m4 = np.load("raw_mode4.npy")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

im0 = axes[0].imshow(m2, cmap='jet')
axes[0].set_title(f"Mode 2 (DTFT) [{m2.min():.1f}, {m2.max():.1f}]")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(m4, cmap='jet')
axes[1].set_title(f"Mode 4 (MATLAB Exact) [{m4.min():.1f}, {m4.max():.1f}]")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(ph_mat, cmap='jet')
axes[2].set_title(f"MATLAB Target [{ph_mat.min():.1f}, {ph_mat.max():.1f}]")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("raw_phase_comparison.png", dpi=150)
print("Saved: raw_phase_comparison.png")

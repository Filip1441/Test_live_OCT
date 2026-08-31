import numpy as np
import h5py
import subprocess
import matplotlib.pyplot as plt

mat = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/phase_schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.mat"
with h5py.File(mat, 'r') as f:
    ph_mat = np.array(f['ph_initial']).T

h, w = ph_mat.shape

# Wrap MATLAB phase
wrapped_mat = np.angle(np.exp(1j * ph_mat)).astype(np.float32)

# Save to binary
wrapped_mat.tofile("test_wrap.bin")

# Compile and run C++ unwrapper
subprocess.run(["clang++", "-std=c++20", "-O3", "tools/test_unwrapper_standalone.cpp", "src/miguel_unwrapper.cpp", "-Iinclude", "-o", "bin/test_unwrapper"])
subprocess.run(["./bin/test_unwrapper", "test_wrap.bin", str(w), str(h), "test_unwrap.bin"])

# Read back
unwrapped_cpp = np.fromfile("test_unwrap.bin", dtype=np.float32).reshape(h, w)

# Align offsets to a background pixel
bg_mat = ph_mat[200, 200]
bg_cpp = unwrapped_cpp[200, 200]

ph_mat_aligned = ph_mat - bg_mat
unwrapped_cpp_aligned = unwrapped_cpp - bg_cpp

diff = unwrapped_cpp_aligned - ph_mat_aligned
rmse = np.sqrt(np.mean(diff**2))

print(f"Standalone C++ unwrapper vs MATLAB unwrapped target -> RMSE: {rmse:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
im0 = axes[0].imshow(unwrapped_cpp_aligned, cmap='jet')
axes[0].set_title("C++ Miguel Unwrapped")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(ph_mat_aligned, cmap='jet')
axes[1].set_title("MATLAB Target")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(diff, cmap='jet')
axes[2].set_title(f"Diff (RMSE={rmse:.2f})")
plt.colorbar(im2, ax=axes[2])
plt.savefig("standalone_unwrap_compare.png", dpi=150)
print("Saved standalone_unwrap_compare.png")

import numpy as np
import h5py
import matplotlib.pyplot as plt

mat = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/phase_schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.mat"
with h5py.File(mat, 'r') as f:
    ph_mat = np.array(f['ph_initial']).T

m2 = np.load("raw_mode2.npy")
m4 = np.load("raw_mode4.npy")

# Calculate wrapped phase
wrap_mat = np.angle(np.exp(1j * ph_mat))
wrap_m2 = np.angle(np.exp(1j * m2))
wrap_m4 = np.angle(np.exp(1j * m4))

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

im0 = axes[0].imshow(wrap_m2, cmap='hsv', vmin=-np.pi, vmax=np.pi)
axes[0].set_title("Wrapped Mode 2")
plt.colorbar(im0, ax=axes[0])

im1 = axes[1].imshow(wrap_m4, cmap='hsv', vmin=-np.pi, vmax=np.pi)
axes[1].set_title("Wrapped Mode 4")
plt.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(wrap_mat, cmap='hsv', vmin=-np.pi, vmax=np.pi)
axes[2].set_title("Wrapped MATLAB Target")
plt.colorbar(im2, ax=axes[2])

plt.tight_layout()
plt.savefig("wrapped_comparison.png", dpi=150)
print("Saved: wrapped_comparison.png")

# Compute difference in wrapped phase directly (after aligning global offset to 0 at a background pixel)
# Pick a background pixel, say (200, 200)
bg_mat = wrap_mat[200, 200]
bg_m4 = wrap_m4[200, 200]

# Shift them so background is 0
wrap_mat_shifted = np.angle(np.exp(1j * (wrap_mat - bg_mat)))
wrap_m4_shifted = np.angle(np.exp(1j * (wrap_m4 - bg_m4)))

# Print center pixel (the small block) phase, say (700, 700)
print(f"Center block wrapped phase (relative to BG): MATLAB={wrap_mat_shifted[700,700]:.2f}, C++ Mode4={wrap_m4_shifted[700,700]:.2f}")

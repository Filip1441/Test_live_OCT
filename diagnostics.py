#!/usr/bin/env python3
"""
Diagnostics: compare MATLAB pipeline (qrOCT.m) step by step with C++ pipeline.
Goal: find where the sign flip / spectral half selection causes the step object to split.
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

obj_path = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.hdf5"
ref_path = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/reference___2026-8-26_16-48-32.hdf5"
mat_path = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/phase_schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.mat"

# Load data
print("[1] Loading HDF5 data...")
with h5py.File(obj_path, 'r') as f:
    obj_data = np.array(f['object']['images']).astype(np.float64)  # (Nlambda, Y, X)
    print(f"    Object shape: {obj_data.shape}")
    
with h5py.File(ref_path, 'r') as f:
    ref_data = np.array(f['reference']['images']).astype(np.float64)
    print(f"    Reference shape: {ref_data.shape}")

# MATLAB reference
with h5py.File(mat_path, 'r') as f:
    ph_matlab = np.array(f['ph_initial']).T.astype(np.float64)
    print(f"    MATLAB ph_initial shape: {ph_matlab.shape}")

Nlambda, Y, X = obj_data.shape
print(f"    Nlambda={Nlambda}, Y={Y}, X={X}")

# ============================================================
# MATLAB pipeline exactly as in qrOCT.m
# ============================================================
print("\n[2] Running MATLAB-exact pipeline in Python...")

# diff_order = 1 (from qrOCT.m line 45: diff_order = 1)
diff_order = 1

# Step 1: IFFT along lambda axis (dim=3 in MATLAB = axis 0 here)
# MATLAB: fftshift(ifft(ifftshift(data), Nz, 3))
print("    IFFT along lambda...")
# ifftshift along lambda axis (axis=0)
data_shifted = np.fft.ifftshift(obj_data, axes=0)
OCTrec_obj = np.fft.fftshift(np.fft.ifft(data_shifted, axis=0), axes=0)

data_shifted = np.fft.ifftshift(ref_data, axes=0)
OCTrec_ref = np.fft.fftshift(np.fft.ifft(data_shifted, axis=0), axes=0)

Nz = OCTrec_obj.shape[0]
print(f"    Nz after IFFT: {Nz}")

# Step 2: Select cross-correlation component
if diff_order == 1:
    z1 = int(round(Nz/2 + 0.07*Nz))
    z2 = Nz
elif diff_order == -1:
    z1 = 0
    z2 = int(round(Nz/2 - 0.07*Nz))

print(f"    diff_order={diff_order}, z1={z1}, z2={z2} (selecting z-window)")

CC_obj = OCTrec_obj[z1:z2, :, :]
CC_ref = OCTrec_ref[z1:z2, :, :]
print(f"    CC_obj shape after z-crop: {CC_obj.shape}")

# Step 3: FFT back to k-space (forward FFT)
# MATLAB: fftshift(fft(ifftshift(CC_obj), Nz_cc, 3))
print("    FFT back to k-space...")
Nz_cc = CC_obj.shape[0]
U_obj = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(CC_obj, axes=0), axis=0), axes=0)
U_ref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(CC_ref, axes=0), axis=0), axes=0)

# Step 4: Complex division U_obj / U_ref
print("    Complex division...")
Z = U_obj / (U_ref + 1e-12)

# Step 5: complexAmp => amp, ph = angle(Z), then rebuild Z = amp*exp(1j*ph) (effectively just keeps phase cleaned)
amp = np.abs(Z)
ph = np.angle(Z)
ph = np.nan_to_num(ph, nan=0.0)

# Step 6: IFFT back to z-space
# MATLAB: fftshift(ifft(ifftshift(Z), Nz_cc, 3))
print("    IFFT back to z-space for CC_corr...")
Z_rebuilt = amp * np.exp(1j * ph)
OCTrec_corr = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Z_rebuilt, axes=0), axis=0), axes=0)

# Step 7: CC_corr - find peak
# MATLAB CC_corr.m: amp_temp = squeeze(amp_corr(:,round(size(amp_corr,2)/2),:)).'; 
#                    [z1, ~] = find(amp_temp == max(amp_temp(:)));
amp_corr = np.abs(OCTrec_corr)
# For simplicity, find global max z-index
amp_sum = np.sum(np.sum(amp_corr, axis=1), axis=1)  # sum over x,y for each z
peak_z = np.argmax(amp_sum)
print(f"    Peak z-index (CC_corr): {peak_z}")

# Also try the exact MATLAB way
mid_col = amp_corr.shape[2] // 2
amp_temp = amp_corr[:, :, mid_col].T  # (X, Nz_cc) transposed to match MATLAB
flat_max_idx = np.unravel_index(np.argmax(amp_temp), amp_temp.shape)
peak_z_matlab = flat_max_idx[1]  # z-index in MATLAB convention 
print(f"    Peak z-index (MATLAB CC_corr way): {peak_z_matlab}")

# Extract phase at peak z
E = OCTrec_corr[peak_z, :, :]  # 2D complex field
ph_end = np.angle(E).astype(np.float32)
print(f"    Phase map shape: {ph_end.shape}")

# Step 8: Unwrap (Miguel) - we skip and just compare wrapped phases
print("\n[3] Comparing with MATLAB result...")

# The MATLAB result might have been unwrapped already
# Let's check correlation of wrapped phase
corr_wrapped = np.corrcoef(ph_end.flatten(), ph_matlab.flatten())[0,1] if ph_end.shape == ph_matlab.shape else 0.0
print(f"    Correlation (wrapped python vs MATLAB ph_initial): {corr_wrapped:.4f}")

# Also try with diff_order = -1
print("\n[4] Now trying diff_order = -1 (other half of spectrum)...")
z1_neg = 0
z2_neg = int(round(Nz/2 - 0.07*Nz))
print(f"    z1={z1_neg}, z2={z2_neg}")

CC_obj_neg = OCTrec_obj[z1_neg:z2_neg, :, :]
CC_ref_neg = OCTrec_ref[z1_neg:z2_neg, :, :]
Nz_cc_neg = CC_obj_neg.shape[0]

U_obj_neg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(CC_obj_neg, axes=0), axis=0), axes=0)
U_ref_neg = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(CC_ref_neg, axes=0), axis=0), axes=0)
Z_neg = U_obj_neg / (U_ref_neg + 1e-12)
amp_neg = np.abs(Z_neg)
ph_neg = np.angle(Z_neg)
ph_neg = np.nan_to_num(ph_neg, nan=0.0)
Z_neg_rebuilt = amp_neg * np.exp(1j * ph_neg)
OCTrec_corr_neg = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(Z_neg_rebuilt, axes=0), axis=0), axes=0)

amp_corr_neg = np.abs(OCTrec_corr_neg)
amp_sum_neg = np.sum(np.sum(amp_corr_neg, axis=1), axis=1)
peak_z_neg = np.argmax(amp_sum_neg)
print(f"    Peak z-index (CC_corr, neg half): {peak_z_neg}")

E_neg = OCTrec_corr_neg[peak_z_neg, :, :]
ph_end_neg = np.angle(E_neg).astype(np.float32)

if ph_end_neg.shape == ph_matlab.shape:
    corr_neg = np.corrcoef(ph_end_neg.flatten(), ph_matlab.flatten())[0,1]
    print(f"    Correlation (neg half vs MATLAB): {corr_neg:.4f}")

# ============================================================
# Compare with C++ output
# ============================================================
print("\n[5] Loading C++ reconstruction result...")
import scipy.io as sio
import os
cpp_path = "output/reconstructed_phase.mat"
if os.path.exists(cpp_path):
    try:
        cpp_data = sio.loadmat(cpp_path)['ph_reconstructed']
    except:
        with h5py.File(cpp_path, 'r') as f:
            cpp_data = np.array(f['ph_reconstructed']).T
    
    if cpp_data.shape == ph_matlab.shape:
        corr_cpp = np.corrcoef(cpp_data.flatten(), ph_matlab.flatten())[0,1]
        print(f"    Correlation (C++ vs MATLAB): {corr_cpp:.4f}")
    
    if cpp_data.shape == ph_end.shape:
        corr_cpp_py = np.corrcoef(cpp_data.flatten(), ph_end.flatten())[0,1]
        print(f"    Correlation (C++ vs Python pos-half): {corr_cpp_py:.4f}")
    
    if cpp_data.shape == ph_end_neg.shape:
        corr_cpp_py_neg = np.corrcoef(cpp_data.flatten(), ph_end_neg.flatten())[0,1]
        print(f"    Correlation (C++ vs Python neg-half): {corr_cpp_py_neg:.4f}")

# ============================================================
# Diagnostic plots
# ============================================================
print("\n[6] Generating diagnostic plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
vmin, vmax = -3.14, 3.14

axes[0,0].set_title("C++ Output (Unwrapped)")
if os.path.exists(cpp_path):
    im = axes[0,0].imshow(cpp_data, cmap='jet')
    plt.colorbar(im, ax=axes[0,0])

axes[0,1].set_title("MATLAB ph_initial")
im = axes[0,1].imshow(ph_matlab, cmap='jet')
plt.colorbar(im, ax=axes[0,1])

axes[0,2].set_title("Python diff_order=+1 (wrapped)")
im = axes[0,2].imshow(ph_end, cmap='jet', vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=axes[0,2])

axes[1,0].set_title("Python diff_order=-1 (wrapped)")
im = axes[1,0].imshow(ph_end_neg, cmap='jet', vmin=vmin, vmax=vmax)
plt.colorbar(im, ax=axes[1,0])

# Z-profile diagnostic
axes[1,1].set_title("Z-profile (amp sum, diff_order=+1)")
axes[1,1].plot(amp_sum)
axes[1,1].axvline(peak_z, color='r', label=f'peak={peak_z}')
axes[1,1].legend()

axes[1,2].set_title("Z-profile (amp sum, diff_order=-1)")
axes[1,2].plot(amp_sum_neg)
axes[1,2].axvline(peak_z_neg, color='r', label=f'peak={peak_z_neg}')
axes[1,2].legend()

plt.tight_layout()
plt.savefig("diagnostics_pipeline.png", dpi=150)
print("    Saved: diagnostics_pipeline.png")
print("\nDone!")

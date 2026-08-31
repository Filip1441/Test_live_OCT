import subprocess
import time
import h5py
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os

obj_hdf5 = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.hdf5"
ref_hdf5 = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/reference___2026-8-26_16-48-32.hdf5"
mat_target = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/phase_schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.mat"

print("==================================================")
print("  Uruchamianie testu potoku C++ vs MATLAB...      ")
print("==================================================")

# 1. Run reconstruction
print("[1] Uruchamianie rekonstrukcji C++ (Tryb 4 - MATLAB Exact bez GUI)...")
start_time = time.time()

process = subprocess.Popen(
    ['./bin/phase_oct_recon', obj_hdf5, ref_hdf5],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True
)

# Inputs:
# 1. Mode: 4 (Matlab Exact bez GUI)
# 2. RI: 1.52
input_str = "4\n1.52\n"
stdout, stderr = process.communicate(input_str)

end_time = time.time()
print(f"----> Rekonstrukcja zajęła: {end_time - start_time:.2f} s")

# Extract compute time from stdout
for line in stdout.split('\n'):
    if "Czas rekonstrukcji" in line:
        print(f"----> {line.strip()}")

# 2. Load C++ Result
cpp_mat_path = "output/reconstructed_phase.mat"
if not os.path.exists(cpp_mat_path):
    print("Error: output/reconstructed_phase.mat not found!")
    exit(1)

cpp_data = sio.loadmat(cpp_mat_path)['ph_reconstructed']
print(f"\n[2] Wczytano wynik C++: {cpp_data.shape}")

# 3. Load MATLAB Target
target_data = None
try:
    target_mat = sio.loadmat(mat_target)
    for k in target_mat.keys():
        if not k.startswith('__') and isinstance(target_mat[k], np.ndarray):
            target_data = target_mat[k]
            break
except Exception as e:
    with h5py.File(mat_target, 'r') as f:
        for k in ['ph_reconstructed', 'ph_initial', 'phase', 'ph', 'data']:
            if k in f:
                target_data = np.array(f[k]).T
                break
        if target_data is None:
            for k in f.keys():
                if k != '#refs#' and hasattr(f[k], 'shape') and len(f[k].shape) >= 2 and max(f[k].shape) > 100:
                    target_data = np.array(f[k]).T
                    break

if target_data is None:
    print("Nie udało się wczytać danych docelowych z pliku MAT.")
    exit(1)

print(f"[3] Wczytano referencje z MATLAB-a (kolezanki): {target_data.shape}")

# 4. Compare
print("\n[4] Szukanie najlepszego dopasowania i liczenie korelacji...")
best_corr = 0
best_target = target_data

transforms = [
    lambda x: x,
    lambda x: np.fliplr(x),
    lambda x: np.flipud(x),
    lambda x: np.flipud(np.fliplr(x)),
    lambda x: x.T,
    lambda x: np.fliplr(x.T),
    lambda x: np.flipud(x.T),
    lambda x: np.flipud(np.fliplr(x.T))
]

for tf in transforms:
    test_tgt = tf(target_data)
    if test_tgt.shape == cpp_data.shape:
        corr = np.corrcoef(cpp_data.flatten(), test_tgt.flatten())[0,1]
        if abs(corr) > abs(best_corr):
            best_corr = corr
            best_target = test_tgt

# Handle phase sign (REMOVED auto-flip to see raw output)
# if best_corr < 0:
#     cpp_data = -cpp_data

# Background alignment (top left corner)
h, w = cpp_data.shape
cpp_bg = np.mean(cpp_data[0:h//4, 0:w//4])
tgt_bg = np.mean(best_target[0:h//4, 0:w//4])

cpp_data_aligned = cpp_data - cpp_bg + tgt_bg
diff = cpp_data_aligned - best_target
rmse = np.sqrt(np.mean(diff**2))

print(f"==================================================")
print(f"  WYNIKI PORÓWNANIA C++ vs MATLAB                 ")
print(f"==================================================")
print(f"  Korelacja (2D): {abs(best_corr)*100:.4f}%")
print(f"  Błąd RMS (RMSE): {rmse:.4f} rad")
print(f"==================================================")

# 5. Plot
print("\n[5] Zapisywanie wykresów do pliku 'unwrapping_comparison.png'...")
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.title("Potok C++ (Miguel 2D Unwrapping)")
plt.imshow(cpp_data_aligned, cmap='jet', vmin=-4, vmax=4)
plt.colorbar()

plt.subplot(132)
plt.title("Skrypt Koleżanki (MATLAB Target)")
plt.imshow(best_target, cmap='jet', vmin=-4, vmax=4)
plt.colorbar()

plt.subplot(133)
plt.title(f"Różnica (RMSE = {rmse:.3f} rad)")
plt.imshow(diff, cmap='jet', vmin=-1, vmax=1)
plt.colorbar()

plt.tight_layout()
plt.savefig("unwrapping_comparison.png", dpi=150)
print("Gotowe! Wykres został zapisany.")

import subprocess
import time
import numpy as np
import scipy.io as sio
import h5py
import os

obj_hdf5 = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.hdf5"
ref_hdf5 = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/reference___2026-8-26_16-48-32.hdf5"
mat_target = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/phase_schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.mat"

def load_target():
    with h5py.File(mat_target, 'r') as f:
        return np.array(f['ph_initial']).T

target_data = load_target()

def run_mode(mode):
    print(f"\n--- Uruchamianie Trybu {mode} ---")
    start_time = time.time()
    
    process = subprocess.Popen(
        ['./bin/phase_oct_recon', obj_hdf5, ref_hdf5],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    input_str = f"{mode}\n1.52\n"
    stdout, stderr = process.communicate(input_str)
    
    compute_time = -1
    for line in stdout.split('\n'):
        if "Czas obliczen (RAM -> Faza 2D)" in line:
            parts = line.split(':')
            compute_time = float(parts[-1].split('ms')[0].strip())
            print(f"[{mode}] Compute Time: {compute_time:.2f} ms")
    
    cpp_data = sio.loadmat("output/reconstructed_phase.mat")['ph_reconstructed']
    
    best_corr = 0
    best_target = target_data
    transforms = [
        lambda x: x, lambda x: np.fliplr(x), lambda x: np.flipud(x), lambda x: np.flipud(np.fliplr(x)),
        lambda x: x.T, lambda x: np.fliplr(x.T), lambda x: np.flipud(x.T), lambda x: np.flipud(np.fliplr(x.T))
    ]
    for tf in transforms:
        test_tgt = tf(target_data)
        if test_tgt.shape == cpp_data.shape:
            corr = np.corrcoef(cpp_data.flatten(), test_tgt.flatten())[0,1]
            if abs(corr) > abs(best_corr):
                best_corr = corr
                best_target = test_tgt
                
    # Align backgrounds
    h, w = cpp_data.shape
    cpp_bg = np.mean(cpp_data[0:h//4, 0:w//4])
    tgt_bg = np.mean(best_target[0:h//4, 0:w//4])
    cpp_aligned = cpp_data - cpp_bg + tgt_bg
    
    rmse = np.sqrt(np.mean((cpp_aligned - best_target)**2))
    
    print(f"[{mode}] Korelacja 2D: {abs(best_corr)*100:.4f}%")
    print(f"[{mode}] RMSE z MATLAB target: {rmse:.4f} rad")
    return compute_time, abs(best_corr), rmse

if __name__ == "__main__":
    t2, c2, r2 = run_mode(2) # Optimized DTFT (Headless)
    t4, c4, r4 = run_mode(4) # Matlab Exact (Headless)
    
    print("\n================ PODSUMOWANIE ================")
    print(f"Tryb 2 (Fast DTFT):  {t2:6.1f} ms | Korelacja: {c2*100:6.2f}% | RMSE: {r2:5.2f} rad")
    print(f"Tryb 4 (Matlab Exact): {t4:6.1f} ms | Korelacja: {c4*100:6.2f}% | RMSE: {r4:5.2f} rad")
    print(f"Przyspieszenie T2 vs T4: {t4/t2:.2f}x")
    print("==============================================")

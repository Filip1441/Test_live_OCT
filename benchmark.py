import subprocess
import time
import os

obj_hdf5 = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/schodek_5um_NA13_prazki_ODT_on_axis_ROI1504_803-878nm_251wavl_1ang_Exp110___2026-8-26_16-46-39.hdf5"
ref_hdf5 = "/Users/filip/Downloads/OCT_calkiem_ok_zestaw/reference___2026-8-26_16-48-32.hdf5"

def run_mode(mode_num, name):
    print(f"==================================================")
    print(f"  Uruchamianie Trybu {mode_num}: {name} ")
    print(f"==================================================")
    
    start_time = time.time()
    
    process = subprocess.Popen(
        ['./bin/phase_oct_recon', obj_hdf5, ref_hdf5],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    input_str = f"{mode_num}\n1.52\n"
    stdout, stderr = process.communicate(input_str)
    
    end_time = time.time()
    total_time = end_time - start_time
    
    recon_time = None
    for line in stdout.split('\n'):
        if "[Unwrap Profiling]" in line:
            print(f"    {line.strip()}")
        if "Czas rekonstrukcji" in line:
            print(f"----> {line.strip()}")
            try:
                # e.g., "Czas rekonstrukcji Fazy: 3.12 s"
                parts = line.strip().split()
                for p in parts:
                    try:
                        recon_time = float(p)
                        break
                    except ValueError:
                        pass
            except:
                pass
    
    print(f"----> Całkowity czas wykonania (z I/O): {total_time:.2f} s\n")
    return recon_time, total_time

if __name__ == "__main__":
    t_recon_fast, t_tot_fast = run_mode(2, "Zoptymalizowane C++ bez GUI (Fast DTFT)")
    t_recon_matlab, t_tot_matlab = run_mode(4, "MATLAB Exact bez GUI (Full FFT)")
    
    print("==================================================")
    print("  PODSUMOWANIE BENCHMARKU                         ")
    print("==================================================")
    if t_recon_fast and t_recon_matlab:
        speedup = t_recon_matlab / t_recon_fast
        print(f"  Zoptymalizowane C++ (DTFT) - Czas rdzenia: {t_recon_fast:.2f} s")
        print(f"  MATLAB Exact (FFT)         - Czas rdzenia: {t_recon_matlab:.2f} s")
        print(f"  ---> Przyspieszenie (tylko obliczenia): {speedup:.2f}x ({(speedup - 1) * 100:.0f}%) szybciej")
    
    speedup_tot = t_tot_matlab / t_tot_fast
    print(f"\n  Zoptymalizowane C++ (DTFT) - Całkowity czas: {t_tot_fast:.2f} s")
    print(f"  MATLAB Exact (FFT)         - Całkowity czas: {t_tot_matlab:.2f} s")
    print(f"  ---> Przyspieszenie całkowite (z IO): {speedup_tot:.2f}x ({(speedup_tot - 1) * 100:.0f}%) szybciej")
    print("==================================================")

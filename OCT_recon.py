#!/Users/filip/Documents/Kod_python/Test_live_OCT/.venv/bin/python
import argparse
import ctypes
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt

try:
    from fair_core import fair_load, fair_edit_metadata, fair_save
except ImportError:
    print("Error: fair_core module not found. Activate the correct virtual environment.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Phase-OCT Full Reconstruction (FAIR integrated)")
    parser.add_argument("--config", type=str, default="/Users/filip/Library/Application Support/LocalViewer/config.json", help="Path to config.json")
    parser.add_argument("--mode", type=str, choices=["fast", "matlab"], default="matlab", help="Pipeline Mode")
    parser.add_argument("--gui", action="store_true", help="Display reconstructed phase map via matplotlib")
    parser.add_argument("--dont_save", action="store_true", help="Skip saving results to FAIR")
    parser.add_argument("--uuid_obj", type=str, help="UUID for Object measurement")
    parser.add_argument("--uuid_ref", type=str, help="UUID for Reference measurement")
    args = parser.parse_args()

    print("--- Phase-OCT Reconstruction Engine ---")

    # 1. Load Object Data
    if args.uuid_obj:
        obj_full = fair_load(args.config, uuid=args.uuid_obj)
        if not obj_full:
            print("Error: Object not found.")
            return
        obj_preview = obj_full
    else:
        obj_preview = fair_load(args.config, ("Data_Type", "raw_projections"), metadata_only=True)
        if not obj_preview:
            return
        obj_full = fair_load(args.config, uuid=obj_preview.metadata["UUID"])

    # 2. Load Reference Data
    if args.uuid_ref:
        ref_full = fair_load(args.config, uuid=args.uuid_ref)
        if not ref_full:
            print("Error: Reference not found.")
            return
        ref_preview = ref_full
    else:
        # Filter by Data_Type and match object's size_z
        filters = [("Data_Type", "raw_projections")]
        obj_size_z = obj_preview.metadata.get("size_z")
        if obj_size_z is not None:
            filters.append(("size_z", obj_size_z))

        ref_preview = fair_load(args.config, *filters, metadata_only=True)
        if not ref_preview:
            return
        ref_full = fair_load(args.config, uuid=ref_preview.metadata["UUID"])

    # 3. Prepare Metadata & Vectors
    parent_uuids = [obj_preview.metadata["UUID"], ref_preview.metadata["UUID"]]
    
    # GUI with fair_edit_metadata (generates new valid UUID and merges parents)
    new_metadata = fair_edit_metadata(parent_uuids, config=args.config)
    if not new_metadata:
        print("Metadata edit cancelled.")
        return

    # Update key fields for reconstruction
    new_metadata["Data_Type"] = "reconstruction"
    base_name = str(obj_full.metadata.get("Measurement_Name", "Unknown"))
    new_metadata["Measurement_Name"] = f"{base_name}_recon"
    new_metadata["Recon_mode"] = f"fair_{args.mode}"
    
    # Remove Creation_Date to let the library generate a new date upon save
    new_metadata.pop("Creation_Date", None)
    new_metadata.pop("creation_date", None)
    
    n_immersion = float(new_metadata.get("n_immersion", 1.52))

    # Extract Numpy matrices
    obj_name = list(obj_full.data.keys())[0]
    ref_name = list(ref_full.data.keys())[0]

    obj_data = np.squeeze(obj_full.data[obj_name])
    ref_data = np.squeeze(ref_full.data[ref_name])

    if obj_data.ndim == 3 and ref_data.ndim == 2:
        # ref_data: (Lambdas, Width), obj_data: (Lambdas, Height, Width)
        ref_data = np.repeat(ref_data[:, np.newaxis, :], obj_data.shape[1], axis=1)

    # Convert to contiguous float32 (Zero-Copy compatible)
    obj_data = np.ascontiguousarray(obj_data, dtype=np.float32)
    ref_data = np.ascontiguousarray(ref_data, dtype=np.float32)

    if obj_data.shape != ref_data.shape:
        print(f"Error: Object shape {obj_data.shape} mismatch with Reference shape {ref_data.shape}")
        return

    num_lambdas, height, width = obj_data.shape
    out_phase = np.zeros((height, width), dtype=np.float32)

    import platform
    system = platform.system()
    if system == "Darwin":
        lib_name = "libphase_oct.dylib"
    elif system == "Windows":
        lib_name = "phase_oct.dll" # CMake on Windows typically creates <name>.dll without 'lib' prefix, or libphase_oct.dll depending on config. We'll check both.
    else:
        lib_name = "libphase_oct.so"

    # Load C++ Shared Library
    lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "bin", lib_name))
    if system == "Windows" and not os.path.exists(lib_path):
        lib_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "bin", "libphase_oct.dll"))
        
    if not os.path.exists(lib_path):
        print(f"Error: C++ engine not found at {lib_path}. Please build the project first.")
        return

    lib = ctypes.CDLL(lib_path)
    lib.run_phase_oct_recon.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_bool,
        ctypes.c_bool,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.run_phase_oct_recon.restype = None

    use_matlab = (args.mode == "matlab")

    # Get RAW Memory Pointers (Releases GIL internally)
    obj_ptr = obj_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    ref_ptr = ref_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    out_ptr = out_phase.ctypes.data_as(ctypes.POINTER(ctypes.c_float))

    start_time = time.time()
    
    print(f"Processing {height}x{width} px, {num_lambdas} waves (Mode: {args.mode.upper()})...")
    # Execute C++ Engine
    lib.run_phase_oct_recon(
        obj_ptr,
        ref_ptr,
        width,
        height,
        num_lambdas,
        n_immersion,
        use_matlab,
        args.gui,
        out_ptr
    )

    compute_time = time.time() - start_time
    new_metadata["time"] = compute_time
    
    print(f"Reconstruction finished in {compute_time * 1000.0:.2f} ms")

    if args.gui:
        plt.figure(figsize=(8, 6))
        plt.imshow(out_phase, cmap="jet", aspect="auto")
        plt.colorbar(label="Phase (rad)")
        plt.title(f"Reconstructed Phase (Mode: {args.mode.upper()})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.show()

    if args.dont_save:
        print("Flag --dont_save active. Skipping FAIR save.")
        return

    result_data = {"ph_reconstructed": out_phase}
    inherited_vectors = obj_full.vectors if obj_full.vectors else None
    
    save_result = fair_save(
        args.config, 
        data=result_data, 
        metadata=new_metadata, 
        vectors=inherited_vectors,
        parent_uuids=parent_uuids
    )

    print(f"Success! Output UUID: {save_result['uuid']}")

if __name__ == "__main__":
    main()

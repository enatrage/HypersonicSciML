import torch
import fenics as fe
import mshr

def verify_installation():
    print("--- Verifying PyTorch & CUDA ---")
    if torch.cuda.is_available():
        print(f"[OK] CUDA is available.")
        print(f"[OK] Device: {torch.cuda.get_device_name(0)}")
        print(f"[OK] PyTorch Version: {torch.__version__}")
    else:
        print("[FAIL] CUDA is NOT available. Check driver/WSL passthrough.")

    print("\n--- Verifying FEniCS Backend ---")
    try:
        # Test standard FEniCS mesh generation
        mesh = fe.UnitSquareMesh(8, 8)
        print(f"[OK] FEniCS initialized. Mesh generated with {mesh.num_cells()} cells.")
        
        # Test mshr boolean operations (often fails if C++ bindings are broken)
        domain = mshr.Rectangle(fe.Point(0, 0), fe.Point(1, 1)) - mshr.Circle(fe.Point(0.5, 0.5), 0.2)
        mesh_mshr = mshr.generate_mesh(domain, 10)
        print(f"[OK] mshr initialized. Boolean mesh generated with {mesh_mshr.num_cells()} cells.")
        
    except Exception as e:
        print(f"[FAIL] FEniCS/mshr verification failed: {e}")

if __name__ == "__main__":
    verify_installation()
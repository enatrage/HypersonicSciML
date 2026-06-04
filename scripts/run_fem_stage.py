import os
import sys
import argparse
import numpy as np

# Append the repository root to the Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from configs.hyperparameters import FEM_PLOT, SNAPSHOT_FILE, T_FINAL, GAMMA_V, NX # The default snapshot relative path defined in hyperparameters.py
from src.fem.solver import run_fem # The main handler function
from src.fem.riemann import exact_riemann # To compute the exact Riemann solution for comparison
from src.utils.plotting import plot_fem_vs_exact # To plot the FEM solution against the
from src.utils.hashing import fem_cache_is_valid, fem_fingerprint # To figure out the cache validity and fingerprint for the FEM settings

def main() -> int:
    p = argparse.ArgumentParser(description="Sod shock tube : SUPG-YZβ FEM generation.")
    p.add_argument("--force-fem", action="store_true", # Flag to force recomputation of FEM even if cache is valid
                   help="Recompute FEM even if cache is valid.")
    p.add_argument("--snapshots", default=SNAPSHOT_FILE, # Defaults to the default snapshot path defined in hyperparameters.py
                   help="Path to the FEM snapshot bundle.")
    args = p.parse_args() # Parses args from the CLI

    print("\n========  STEP 1 : FEM  ========")
    if args.force_fem: # If forcing FEM recomputation, ignore cache and run FEM directly
        print("[INFO] --force-fem set: recomputing FEM.")
        run_fem(snapshot_path=args.snapshots)
    elif fem_cache_is_valid(args.snapshots): # If not forcing but cache is valid, skip FEM and print info
        print(f"[INFO] FEM cache hit: {args.snapshots} "
              f"matches current settings (fingerprint={fem_fingerprint()}). "
              f"Skipping FEM.")
    else: # If not forcing and cache is invalid, run FEM and print info
        if os.path.exists(args.snapshots): # If cache file exists but is invalid, print that it's being recomputed
            print(f"[INFO] FEM cache present at {args.snapshots} but "
                  f"fingerprint differs; recomputing.")
        else: # If cache file doesn't exist, print that it's being computed
            print(f"[INFO] No FEM cache at {args.snapshots}; computing.")
        run_fem(snapshot_path=args.snapshots)
        plot_fem_vs_exact()
        print(f"[INFO] FEM solution computed and plotted against exact solution. "
              f"Plot saved to {FEM_PLOT}.")
        
    return 0 # Exit with success

if __name__ == "__main__":
    sys.exit(main()) # Run the main function and exit with its return code
import os
import sys
import argparse
from dataclasses import fields

# Append the repository root to the Python path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.append(repo_root)

from configs.hyperparameters import TrainConfig, SNAPSHOT_FILE, PINN_CACHE, PASSC_PLOT
from src.training.engine import run_pinn

_CLI_SCALAR_TYPES = (int, float, str)
_CLI_SCALAR_TYPE_MAP = {"int": int, "float": float, "str": str}

def _is_cli_scalar(f) -> bool:
    if isinstance(f.type, type):
        return f.type in _CLI_SCALAR_TYPES
    return f.type in _CLI_SCALAR_TYPE_MAP

def _resolve_field_type(f):
    if isinstance(f.type, type):
        return f.type
    return _CLI_SCALAR_TYPE_MAP[f.type]

def _add_trainconfig_args(parser: argparse.ArgumentParser) -> None:
    for f in fields(TrainConfig):
        if not _is_cli_scalar(f):
            continue
        flag    = "--" + f.name.replace("_", "-")
        ftype   = _resolve_field_type(f)
        help_md = f.metadata.get("help", "")
        parser.add_argument(
            flag, type=ftype, default=f.default, dest=f.name,
            help=f"{help_md} (default: {f.default})"
        )

def _trainconfig_from_args(args: argparse.Namespace) -> TrainConfig:
    kwargs = {f.name: getattr(args, f.name)
              for f in fields(TrainConfig)
              if _is_cli_scalar(f) and hasattr(args, f.name)}
    return TrainConfig(**kwargs)

def main() -> int:
    p = argparse.ArgumentParser(description="Sod shock tube : PASSC-Transient PINN.")
    p.add_argument("--force-pinn", action="store_true",
                   help="Retrain PINN even if cache is valid.")
    p.add_argument("--snapshots", default=SNAPSHOT_FILE,
                   help="Path to the FEM snapshot bundle.")
    p.add_argument("--pinn-cache", default=PINN_CACHE,
                   help="Path to the PINN weight cache.")
    p.add_argument("--device",    default="auto", choices=("auto", "cpu", "cuda"),
                   help="Compute device for the PINN step.")
    p.add_argument("--out",       default=PASSC_PLOT,
                   help="Output plot path for the PINN comparison.")

    _add_trainconfig_args(p)
    args = p.parse_args()

    print("\n========  STEP 2 : PINN  ========")
    cfg = _trainconfig_from_args(args)
    run_pinn(snapshot_path=args.snapshots, cfg=cfg,
             device_str=args.device, out_plot=args.out,
             cache_path=args.pinn_cache,
             force_retrain=args.force_pinn)
             
    return 0

if __name__ == "__main__":
    sys.exit(main())
import argparse
import os
import sys

from src.master.config_parser import build_and_validate_config
from src.master.orchestrator import orchestrate_fem, orchestrate_model

def main():

    # Append the repository root to the Python path, useful downstream
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.append(repo_root)

    # Defines the CLI commands and their interactions
    parser = argparse.ArgumentParser(description="Master script")
    parser.add_argument("-bc", "--base-cfg", action="store", required=True)
    parser.add_argument("-ac", "--addon-cfg", action="store")
    parser.add_argument("-ff", "--force-fem", action="store_true")
    parser.add_argument("-fm", "--force-model", action="store_true")
    args = parser.parse_args()

    base_path = args.base_cfg; addon_path = args.addon_cfg; force_fem = args.force_fem; force_model = args.force_model

    master_cfg = build_and_validate_config(
        base_path=base_path, addon_path=addon_path
    )

    run_fem = master_cfg.run_mode.fem; run_model = master_cfg.run_mode.model

    orchestrate_fem(
        cfg_fem=master_cfg.fem_config, run_fem=run_fem, force_fem=force_fem
    )

    orchestrate_model(
        cfg_model=master_cfg.model_config, cfg_fem=master_cfg.fem_config, run_model=run_model, force_model=force_model
    )

if __name__ == "__main__":
    main()

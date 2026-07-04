from pathlib import Path
import logging

def check_fem_existence(snapshot_path: str, plot_path: str, force_fem: bool) -> bool:
    """
    Checks the existence of FEM files and returns a decision on whether to continue

    Args:
        snapshot_path (str): The path for the snapshot being saved
        plot_path (str): The path where the plot comparison is saved

    Returns:
        bool: Decision on continuing with further FEM calculations or skipping to model
    """
    paths_to_check = [snapshot_path, plot_path]
    existing_paths = [path for path in paths_to_check if Path(path).exists()]
    
    if existing_paths:
        existing_str = ", ".join(existing_paths)
        if not force_fem: 
            logging.warning(f"ORCHESTRATOR: FEM file(s) already exist at: {existing_str}, set force flag to overwrite if needed")
            return False
        logging.warning(f"ORCHESTRATOR: Force flag is True, overwriting existing FEM file(s) at: {existing_str}")
        return True
    
    else:
        return True

def check_model_existence(local_log_path: str, model_save_path: str, plot_comp_path: str, force_model: bool):
    paths_to_check = [local_log_path, model_save_path, plot_comp_path]
    existing_paths = [path for path in paths_to_check if Path(path).exists()]

    if existing_paths:
        existing_str = ", ".join(existing_paths)
        if not force_model:
            logging.warning(f"ORCHESTRATOR: Model file(s) already exist at: {existing_str}, set force flag to overwrite if needed.")
            return False
        logging.warning(f"ORCHESTRATOR: Force flag is True, overwriting existing model file(s) at: {existing_str}.")
        return True
    
    else:
        return True
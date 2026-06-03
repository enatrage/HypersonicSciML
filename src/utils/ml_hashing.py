import os
import torch

from dataclasses import fields, asdict
from .hashing import fingerprint
from configs.hyperparameters import TrainConfig 

def trainconfig_to_dict(cfg: TrainConfig) -> dict:
    """
    Converts a TrainConfig dataclass instance into a dictionary, handling nested dataclasses
    (like the schedule) appropriately.

    Args:
        cfg (TrainConfig): The TrainConfig instance to convert.
    
    Returns:
        dict: A dictionary representation of the TrainConfig instance.
    """
    d = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name == "schedule":
            d["schedule"] = [asdict(ph) for ph in v]
        else:
            d[f.name] = v
    return d

def pinn_fingerprint(cfg: TrainConfig, fem_fp: str) -> str:
    """
    Computes the PINN fingerprint by hashing a dictionary that includes the FEM fingerprint
    and the TrainConfig parameters.

    Args:
        cfg (TrainConfig): The training configuration to include in the fingerprint.
        fem_fp (str): The FEM fingerprint to include in the PINN fingerprint.

    Returns:
        str: A 16-character hexadecimal string representing the PINN fingerprint.
    """
    payload = {"fem_fingerprint": fem_fp, "train_config": trainconfig_to_dict(cfg)}
    return fingerprint(payload)

def pinn_cache_is_valid(cache_path: str, target_fp: str) -> bool:
    """
    Checks if the PINN cache file exists and contains a matching fingerprint.

    Args:
        cache_path (str): The path to the PINN cache file to check.
        target_fp (str): The target fingerprint that the cache should match.

    Returns:
        bool: True if the cache is valid, False otherwise.
    """
    if not os.path.exists(cache_path): return False
    try:
        ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
        return ckpt.get("pinn_fingerprint", "") == target_fp
    except Exception:
        return False
import hashlib
import json
import os
import numpy as np
import torch
from dataclasses import fields, asdict
from configs.hyperparameters import (
    TrainConfig, GAMMA_V, RHO_L_V, U_L_V, P_L_V, 
    RHO_R_V, U_R_V, P_R_V, X_DIAPH, T_FINAL, 
    DT, NX, BETA_V, K_S, U1_REF_V, U2_REF_V, U3_REF_V
)

def fingerprint(d: dict) -> str:
    """
    Gets a dictionary, serializes it to JSON with sorted keys, encodes to UTF-8,
    and returns the first 16 chars of its SHA-1 hash as a fingerprint string.

    Args:
        d (dict): The input dictionary to fingerprint.
    
    Returns:
        str: A 16-character hexadecimal string representing the fingerprint.
    """
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]

def fem_inputs() -> dict:
    """
    Collects the FEM input parameters from the hyperparameters module and returns
    them as a dictionary. These parameters are used to compute the FEM fingerprint.
    """
    return dict(
        GAMMA_V=GAMMA_V, RHO_L_V=RHO_L_V, U_L_V=U_L_V, P_L_V=P_L_V,
        RHO_R_V=RHO_R_V, U_R_V=U_R_V, P_R_V=P_R_V, X_DIAPH=X_DIAPH,
        T_FINAL=T_FINAL, DT=DT, NX=NX, BETA_V=BETA_V, K_S=K_S,
        U1_REF_V=U1_REF_V, U2_REF_V=U2_REF_V, U3_REF_V=U3_REF_V,
    )

def fem_fingerprint() -> str:
    """
    Computes the FEM fingerprint by hashing the dictionary of FEM input parameters.
    """
    return fingerprint(fem_inputs())

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

def fem_cache_is_valid(snapshot_path: str) -> bool:
    """
    Checks if the FEM cache file exists and contains a matching fingerprint.

    Args:
        snapshot_path (str): The path to the FEM snapshot file to check.

    Returns:
        bool: True if the cache is valid, False otherwise.
    """
    if not os.path.exists(snapshot_path): return False
    try:
        with np.load(snapshot_path, allow_pickle=False) as d:
            if "fem_fingerprint" not in d.files: return False
            stored = str(d["fem_fingerprint"])
    except Exception:
        return False
    return stored == fem_fingerprint()

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
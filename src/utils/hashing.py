import hashlib
import json
import os
import numpy as np
from configs.hyperparameters import (
    GAMMA_V, RHO_L_V, U_L_V, P_L_V, 
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


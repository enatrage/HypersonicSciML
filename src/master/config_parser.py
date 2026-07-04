from omegaconf import OmegaConf
from typing import Optional
from .schemas import MasterConfig

def build_and_validate_config(
        base_path: str,
        addon_path: Optional[str] = None
) -> MasterConfig:
    
    # Ingests the base temp
    cfg = OmegaConf.load(base_path)

    # Merges addon confg
    if addon_path is not None:
        addon_cfg = OmegaConf.load(addon_path)
        cfg = OmegaConf.merge(cfg, addon_cfg)

    # Resolve interpolations and convert to standard Python dictionary
    resolved_dict = OmegaConf.to_container(cfg, resolve=True)
    
    # Strict Initialization and Validation, to throw explicit Pydantic ValidationError if keys are missing or types mismatch
    validated_config = MasterConfig(**resolved_dict)
    
    return validated_config


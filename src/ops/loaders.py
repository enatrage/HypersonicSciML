"""Dynamic component loaders (registry/factory pattern) for the training pipeline."""
from typing import Any, Dict, Iterable

import torch

from src.model.pinn import PINN_Euler
from src.training.loss import Arc_Loss

# --- Registries: the ONLY place concrete classes are referenced -------------
MODEL_REGISTRY: Dict[str, type] = {
    "PINN_Euler": PINN_Euler,
}
OPTIM_REGISTRY: Dict[str, type] = {
    "AdamW": torch.optim.AdamW,
    "Adam": torch.optim.Adam,
    "SGD": torch.optim.SGD,
}
SCHEDULER_REGISTRY: Dict[str, type] = {
    "ReduceLROnPlateau": torch.optim.lr_scheduler.ReduceLROnPlateau,
    "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
}
LOSS_REGISTRY: Dict[str, type] = {
    "Arc_Loss": Arc_Loss,
}


def _resolve(registry: Dict[str, type], setup: Any, kind: str) -> type:
    """Look up setup['type'] in registry; raise ValueError before instantiation."""
    if not isinstance(setup, dict) or "type" not in setup:
        raise ValueError(
            f"{kind} setup must be a dict with a 'type' key; got {setup!r}"
        )
    type_name = setup["type"]
    if type_name not in registry:
        raise ValueError(
            f"Unknown {kind} type {type_name!r}. "
            f"Registered {kind}s: {sorted(registry)}"
        )
    return registry[type_name]


def build_model(model_setup: Dict[str, Any], device: torch.device):
    cls = _resolve(MODEL_REGISTRY, model_setup, "model")
    cfg = model_setup.get("cfg_model", {})
    return cls(**cfg).to(device)


def build_optimizer(params: Iterable, optim_setup: Dict[str, Any]):
    cls = _resolve(OPTIM_REGISTRY, optim_setup, "optimizer")
    cfg = optim_setup.get("cfg_optim", {})
    return cls(params, **cfg)


def build_scheduler(optimizer, scheduler_setup: Dict[str, Any]):
    cls = _resolve(SCHEDULER_REGISTRY, scheduler_setup, "scheduler")
    cfg = scheduler_setup.get("cfg_scheduler", {})
    return cls(optimizer, **cfg)


def build_loss(loss_setup: Dict[str, Any], device: torch.device, **runtime_kwargs):
    """cfg_loss holds static kwargs; runtime_kwargs injects dataset-derived tensors."""
    cls = _resolve(LOSS_REGISTRY, loss_setup, "loss")
    cfg = {**loss_setup.get("cfg_loss", {}), **runtime_kwargs}
    return cls(**cfg).to(device)

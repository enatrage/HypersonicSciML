import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from src.master.schemas import ArchitectureConfig, LossConfig, OptimizerConfig, LRSchedConfig

import torch.nn as nn

from src.model.training.data import FemSnapshots
from src.model.loss.losses import CheatLoss

def build_architecture(config: ArchitectureConfig) -> nn.Module:
    """Instantiates the PyTorch network based on the configs"""
    kwargs = config.model_dump(exclude={"type"})
    if config.type == "old_pinn":
        from src.model.architectures.old_pinn import PINN_Euler
        return PINN_Euler(**kwargs)
    elif config.type == "LAIrResPINN":
        from src.model.architectures.la_irrespinn import LA_IrResPINN
        return LA_IrResPINN(**kwargs)

    raise ValueError(f"Architecture type '{config.type}' is not registered in the architecture load factory")

def build_loss(config: LossConfig, model: nn.Module, data: FemSnapshots) -> nn.Module:
    """Instantiates the Loss based on the configs"""
    kwargs = config.model_dump(exclude={"type"})
    if config.type == "cheat_loss":
        from ..loss.losses import CheatLoss
        return CheatLoss(model=model, smooth_map=data.smooth_map, Y_inv=data.Y_inv, 
                         t_axis=data.t_axis, x_axis=data.x_axis, gamma=data.gamma, 
                         t_start=data.t_start, t_final=data.t_final, **kwargs)
        
    raise ValueError(f"Loss type '{config.type}' is not registered in the loss load factory")

def build_optim(config: OptimizerConfig, model: nn.Module) -> optim:
    """Instantiates the PyTorch optimizer based on the configs and model"""
    kwargs = config.model_dump(exclude={"type"})
    if config.type == "adam_w":
        return optim.AdamW(params=model.parameters(), **kwargs)
        
    elif config.type == "adam":
        return optim.Adam(params=model.parameters(), **kwargs)
        
    elif config.type == "sgd":
        return optim.SGD(params=model.parameters(), **kwargs)
        
    elif config.type == "lbfgs":
        return optim.LBFGS(params=model.parameters(), **kwargs)
        
    raise ValueError(f"Optimizer type '{config.type}' is not registered in the optimizer load factory")

def build_scheduler(config: LRSchedConfig, optimizer: optim) -> lr_sched:
    """Instantiates the PyTorch learning rate scheduler based on the config."""
    kwargs = config.model_dump(exclude={"type"}, exclude_none=True)
    if config.type == "reduce_on_plateau":
        return lr_sched.ReduceLROnPlateau(optimizer, **kwargs)
        
    elif config.type == "cos_annealing":
        return lr_sched.CosineAnnealingLR(optimizer, **kwargs)
        
    elif config.type == "cos_ann_warm_restarts":
        return lr_sched.CosineAnnealingWarmRestarts(optimizer, **kwargs)
        
    elif config.type == "one_cycle":
        return lr_sched.OneCycleLR(optimizer, **kwargs)
        
    elif config.type == "step":
        return lr_sched.StepLR(optimizer, **kwargs)
        
    raise ValueError(f"Scheduler type '{config.type}' is not registered in the scheduler load factory.")

def build_model(
        arc_cfg: ArchitectureConfig, 
        loss_cfg: LossConfig, 
        optim_cfg: OptimizerConfig,
        lrsched_cfg: LRSchedConfig,
        data: FemSnapshots
) -> dict:
    model = build_architecture(arc_cfg)
    loss_fn = build_loss(loss_cfg, model, data)
    optimizer = build_optim(optim_cfg, model)
    lr_scheduler = build_scheduler(lrsched_cfg, optimizer)
    return model, loss_fn, optimizer, lr_scheduler




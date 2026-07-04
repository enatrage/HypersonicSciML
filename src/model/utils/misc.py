from ..lrsched.schemas import STEP_SCHEDULERS
import torch.optim.lr_scheduler as lr_sched

import logging
import torch

def check_lrtype_ifstep(scheduler: lr_sched) -> bool:
    return isinstance(scheduler, STEP_SCHEDULERS)

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp

def set_device(device_option):
    assert device_option in ["auto", "cuda", "cpu"], f"Device: {device_option} not in [auto, cuda, cpu]"
    if device_option == "auto":
        if torch.cuda.is_available(): 
            device = torch.device("cuda")
            logging.info("Model: Device set to CUDA")
        else: 
            device = torch.device("cpu")
            logging.info("Model: Device set to 'auto', but set to CPU")
    elif device_option == "cuda":
        device = torch.device("cuda")
        logging.info("Model: Device set to CUDA")
    else:
        device = torch.device("cpu")
        logging.info("Model: Device set to CPU")
    return device

def build_physics_package(U_pred_test, gamma):
    """Builds the physics package for the loss function"""
    rho = U_pred_test[:,0]
    q   = U_pred_test[:,1]
    E   = U_pred_test[:,2]
    u   = q / torch.maximum(rho, torch.tensor(1e-12, device=rho.device))
    p   = (gamma - 1.0) * (E - 0.5 * rho * u**2)
    e   = p / ((gamma - 1.0) * torch.maximum(rho, torch.tensor(1e-12, device=rho.device)))
    return rho, u, p, e

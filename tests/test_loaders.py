import pytest
import torch
import torch.nn as nn

from src.ops.loaders import (
    build_model, build_optimizer, build_scheduler, MODEL_REGISTRY,
)


def test_build_optimizer_unpacks_cfg():
    params = nn.Linear(2, 2).parameters()
    setup = {"type": "AdamW", "cfg_optim": {"lr": 1e-3, "weight_decay": 1e-5}}
    opt = build_optimizer(params, setup)
    assert isinstance(opt, torch.optim.AdamW)
    assert opt.param_groups[0]["lr"] == 1e-3
    assert opt.param_groups[0]["weight_decay"] == 1e-5


def test_build_scheduler_wraps_optimizer():
    opt = torch.optim.SGD(nn.Linear(2, 2).parameters(), lr=0.1)
    setup = {"type": "ReduceLROnPlateau", "cfg_scheduler": {"factor": 0.5}}
    sched = build_scheduler(opt, setup)
    assert isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau)


def test_unknown_type_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown optimizer"):
        build_optimizer(nn.Linear(2, 2).parameters(),
                        {"type": "NotAnOptimizer", "cfg_optim": {}})


def test_missing_type_key_raises_valueerror():
    with pytest.raises(ValueError, match="must be a dict"):
        build_optimizer(nn.Linear(2, 2).parameters(), {"cfg_optim": {}})


def test_build_model_instantiates_registered_class():
    setup = {"type": "PINN_Euler",
             "cfg_model": {"n_hidden": 8, "n_blocks": 1, "n_fourier": 4, "sigma": 1.0}}
    model = build_model(setup, torch.device("cpu"))
    assert isinstance(model, MODEL_REGISTRY["PINN_Euler"])

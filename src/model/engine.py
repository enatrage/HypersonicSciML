from src.master.schemas import ModelConfig, FemConfig

from src.model.utils.logging import setup_wandb, setup_logging
from src.model.training.data import get_data
from src.model.utils.factory import build_model
from src.model.training.training import train_model
from src.model.utils.misc import set_device

import logging
import traceback

import numpy as np
import torch

def model_runner(cfg_model: ModelConfig, cfg_fem: FemConfig):

    # Setup wandb and logging
    setup_logging(local_log_path=cfg_model.export.local_log_path)
    wandb_run = setup_wandb(
        model_config=cfg_model, fem_config=cfg_fem, wandb_entity=cfg_model.export.wandb_entity, 
        wandb_project=cfg_model.export.wandb_project, wandb_name=cfg_model.export.wandb_name
    )
    logging.log("Model: Logging and WandB setup and initialized")

    # Set seed before everything
    np.random.seed(cfg_model.training.seed)
    torch.manual_seed(cfg_model.training.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg_model.training.seed)
    # Set the device
    device = set_device(cfg_model.training.device)

    # Get the data
    data_iterator = get_data(
        snapshot_path=cfg_model.data.snapshot_path, U1_ref=cfg_fem.u1_ref, U2_ref=cfg_fem.u2_ref, U3_ref=cfg_fem.u3_ref,
        batch_size=cfg_model.data.batch_size, grad_quantile=cfg_model.data.grad_quantile, num_workers=cfg_model.data.num_workers,
        pin_memory=cfg_model.data.pin_memory
    )
    logging.log("Model: Data figured out")

    # Get the model, loss, optim, lr_sched, etc.
    model, loss_fn, optimizer, lr_scheduler = build_model(
        arc_cfg=cfg_model.architecture,
        loss_cfg=cfg_model.loss,
        optim_cfg=cfg_model.optimizer,
        lrsched_cfg=cfg_model.lrsched,
        data=data_iterator.dataset
    )
    logging.log("Model: The model, loss, optimizer, scheduler built")

    try:
        logging.info("Model: Training function called")
        train_model(
            model=model, loss_fn=loss_fn, optimizer=optimizer, lr_scheduler=lr_scheduler,
            data_iterator=data_iterator, device=device,
            train_cfg=cfg_model.training, data_cfg=cfg_model.data, export_cfg=cfg_model.export, fem_cfg=cfg_fem
        )
        logging.info("Model: Training function successfully exited")

    except Exception as e:
        wandb_run.alert(title= "Model: Training crashed", text=str(e))
        traceback.print_exc()
        raise e

    finally:
        wandb_run.finish()    

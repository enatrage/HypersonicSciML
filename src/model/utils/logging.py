from src.master.schemas import NetConfig, FemConfig

import logging
import sys
import wandb

def setup_wandb(model_config: NetConfig, fem_config: FemConfig, mode: str, wandb_entity: str, wandb_project: str, wandb_name: str):

    settings = wandb.Settings(
        show_errors=True,
        silent=False,
        show_warnings=True
    )

    # Dumping the configs to dicts so that we can put them up to wandb
    model_dict = model_config.model_dump(mode='json')
    fem_dict = fem_config.model_dump(mode='json')
    combined_config = { # merge them so that we upload it to wandb
        "model": model_dict,
        "fem": fem_dict
    }

    run = wandb.init(
        mode= mode,
        entity= wandb_entity,
        project= wandb_project,
        name= wandb_name,
        config= combined_config,
        job_type= 'model_training',
        settings=settings
    )

    return run

def setup_logging(local_log_path: str):

    if not local_log_path.endswith('.txt'): local_log_path += '.txt'

    logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    force=True,
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(local_log_path, mode='w'), 
        logging.StreamHandler(sys.stdout) # Keeps console output active for W&B
        ]
    )
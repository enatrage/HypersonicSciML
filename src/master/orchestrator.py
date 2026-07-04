import logging

from src.master.schemas import FemConfig, NetConfig
from src.master.check_existence import check_fem_existence, check_model_existence
from src.fem.engine import fem_runner
from src.model.engine import model_runner


def orchestrate_fem(cfg_fem: FemConfig, run_fem: bool, force_fem: bool) -> None:
    """
    Orchestrates the FEM runtime, based on config and run/force commands

    Args:
        cfg_fem (FemConfig): The agreed upon schema for FEM configs
        run_fem (bool): Bool to either run or pass FEM stage
        force_fem (bool): Bool to force FEM stage even if the files specified in config exist

    Returns:
        None: All outputs are handled internally via writing
    """

    if run_fem:
        continue_bool = check_fem_existence(
            snapshot_path=cfg_fem.io.snapshot_path, 
            plot_path=cfg_fem.io.plot_path, 
            force_fem=force_fem
        ) # Checks the existence of previous runs with these path configs
        if continue_bool:
            fem_runner(cfg_fem)

    else:
        logging.info("ORCHESTRATOR: FEM run was turned off, proceeding")


def orchestrate_model(cfg_net: NetConfig, cfg_fem: FemConfig, run_model: bool, force_model: bool) -> None:
    """
    Orchestrates the Model runtime, based on model/fem configs and run/force commands

    Args:
        cfg_net (NetConfig): The agreed upon schema for Network configs
        cfg_fem (FemConfig): The agreed upon schema for FEM configs
        run_model (bool): Bool to either run or pass the Model stage
        force_model (bool): Bool to force Model stage even if the files specified in the config exist

    Returns:
        None: All outputs are handled internally via writing
    """

    if run_model:
        continue_bool = check_model_existence(
            local_log_path=cfg_net.export.local_log_path,
            model_save_path=cfg_net.export.model_save_path,
            plot_comp_path=cfg_net.export.plot_comp_path,
            force_model=force_model
        ) # Checks the existence of previous runs with these path configs
        if continue_bool:
            model_runner(cfg_net, cfg_fem)

    else:
        logging.info("ORCHESTRATOR: Model run was turned off, proceeding")

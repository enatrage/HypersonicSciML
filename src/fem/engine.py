import logging

from src.master.schemas import FemConfig
from src.fem.plotting import plot_fem_vs_exact
from src.fem.solver import run_supg_yzb



def fem_runner(cfg: FemConfig):
    """
    The main running function for the 
    
    """

    ## Unpack the configs first
    rho_L_v = cfg.initial_conditions.left.rho; u_L_v = cfg.initial_conditions.left.u; p_L_v = cfg.initial_conditions.left.p
    rho_R_v = cfg.initial_conditions.right.rho; u_R_v = cfg.initial_conditions.right.u; p_R_v = cfg.initial_conditions.right.p
    # Thermodynamics
    gamma_v = cfg.thermodynamics.gamma
    # Domain
    x_diaph = cfg.domain.x_diaph; nx = cfg.domain.nx; t_final = cfg.domain.t_final; dt = cfg.domain.dt
    # Ref
    U1_ref_v = cfg.u1_ref; U2_ref_v = cfg.u2_ref; U3_ref_v = cfg.u3_ref
    # Scaling
    beta_v = cfg.scaling.beta; k_s = cfg.scaling.k_s
    # IO
    snapshot_path = cfg.io.snapshot_path; plot_path = cfg.io.plot_path

    # Runs the YZB stabilized SUPG and saves the created snapshot
    run_supg_yzb(
        rho_L_v=rho_L_v, u_L_v=u_L_v, p_L_v=p_L_v, rho_R_v=rho_R_v, u_R_v=u_R_v, p_R_v=p_R_v,
        gamma_v=gamma_v, x_diaph=x_diaph, nx=nx, t_final=t_final, dt=dt, 
        U1_ref_v=U1_ref_v, U2_ref_v=U2_ref_v, U3_ref_v=U3_ref_v, beta_v=beta_v, k_s=k_s, snapshot_path=snapshot_path
    )

    # Plots (SUPG-YZB vs exact Riemann) and saves
    plot_fem_vs_exact(
        gamma_v=gamma_v, rho_L_v=rho_L_v, u_L_v=u_L_v, p_L_v=p_L_v,
        rho_R_v=rho_R_v, u_R_v=u_R_v, p_R_v=p_R_v, x_diaph=x_diaph,
        t_final=t_final, nx=nx, plot_path=plot_path, snapshot_path=snapshot_path
    )
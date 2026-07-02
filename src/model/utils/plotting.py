import matplotlib.pyplot as plt
import wandb
import numpy as np
import torch
from typing import Tuple

from src.fem.riemann import exact_riemann
from src.master.schemas import FemConfig

def plot_pred_comparison(
    test_plot_package: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    x_points: torch.Tensor, plot_comp_path: str, snapshot_path: str, fem_cfg: FemConfig
) -> None:

    ## Unpack the configs first
    rho_L = fem_cfg.initial_conditions.left.rho; u_L = fem_cfg.initial_conditions.left.u; p_L = fem_cfg.initial_conditions.left.p
    rho_R = fem_cfg.initial_conditions.right.rho; u_R = fem_cfg.initial_conditions.right.u; p_R = fem_cfg.initial_conditions.right.p
    # Thermodynamics and domain
    gamma = fem_cfg.thermodynamics.gamma; x_diaph = fem_cfg.domain.x_diaph; t_final = fem_cfg.domain.t_final

    rho_ex, u_ex, p_ex = exact_riemann(
        x_points, t_final, gamma, rho_L, u_L, p_L, rho_R, u_R, p_R, x_diaph
    )
    e_ex = p_ex / ((gamma - 1.0) * rho_ex)
    fem_data = np.load(snapshot_path)
    
    rho_snap = fem_data['rho_snap'] # Assuming the snapshot file contains 'rho_snap', 'q_snap', and 'E_snap' arrays
    q_snap   = fem_data['q_snap']
    E_snap   = fem_data['E_snap']
    q_fem   = q_snap[-1]
    E_fem   = E_snap[-1]

    rho_fem = rho_snap[-1]
    u_fem   = q_fem / rho_fem
    p_fem = (gamma - 1.0) * (E_fem - 0.5 * rho_fem * u_fem**2)
    e_fem = p_fem / ((gamma - 1.0) * rho_fem)

    ex_package = (rho_ex, u_ex, p_ex, e_ex)
    fem_package = (rho_fem, u_fem, p_fem, e_fem)
    
    _plot_pinn_vs_fem_vs_exact(x_points, ex_package, fem_package, test_plot_package, plot_comp_path)
    pass

def _plot_pinn_vs_fem_vs_exact(
    x: torch.Tensor, 
    exact: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    fem: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    pinn: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    save_path: str
) -> None:
    """
    Plots the 2x2 grid comparing the PASSC-Transient PINN against both the 
    FEM baseline and the exact Riemann solution.
    Expects tuples of (rho, u, p, e).
    """
    rho_ex, u_ex, p_ex, e_ex = exact
    rho_fem, u_fem, p_fem, e_fem = fem
    rho_p, u_p, p_p, e_p = pinn

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
    panels = [
        (ax[0,0], rho_ex, rho_fem, rho_p, r'$\rho$', 'Density'),
        (ax[0,1], u_ex,   u_fem,   u_p,   'u',       'Velocity'),
        (ax[1,0], p_ex,   p_fem,   p_p,   'p',       'Pressure'),
        (ax[1,1], e_ex,   e_fem,   e_p,   'e',       'Specific internal energy')
    ]

    for axi, ex, fe, pi, ylab, title in panels:
        axi.plot(x, ex, 'k-',  lw=1.5, label='Exact')
        axi.plot(x, fe, 'g--', lw=1.0, label=r'SUPG-YZ$\beta$')
        axi.plot(x, pi, 'r-',  lw=1.0, label='PASSC')
        axi.set_xlabel('x')
        axi.set_ylabel(ylab)
        axi.set_title(title)
        axi.legend()
        axi.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    wandb.log({"general/Riemann_Solution_Comparison": wandb.Image(save_path)})
    plt.close(fig)



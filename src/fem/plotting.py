import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple

from src.fem.riemann import exact_riemann

def plot_fem_vs_exact(gamma_v, rho_L_v, u_L_v, p_L_v, rho_R_v, u_R_v, p_R_v, x_diaph, t_final, nx, plot_path, snapshot_path) -> None: 
    
    xpoints = np.linspace(0.0, 1.0, nx+1) # Number of nodes +1 than number of cells
    rho_ex, u_ex, p_ex = exact_riemann(xpoints, t_final, gamma_v, rho_L_v, u_L_v, p_L_v, rho_R_v, u_R_v, p_R_v, x_diaph)
    e_ex = p_ex / ((gamma_v - 1.0) * rho_ex)
    fem_data = np.load(snapshot_path)
    
    rho_snap = fem_data['rho_snap'] # Assuming the snapshot file contains 'rho_snap', 'q_snap', and 'E_snap' arrays
    q_snap   = fem_data['q_snap']
    E_snap   = fem_data['E_snap']
    rho_fem = rho_snap[-1]
    q_fem   = q_snap[-1]
    E_fem   = E_snap[-1]

    u_fem   = q_fem / rho_fem
    p_fem = (gamma_v - 1.0) * (E_fem - 0.5 * rho_fem * u_fem**2)
    e_fem = p_fem / ((gamma_v - 1.0) * rho_fem)
    # scripts/run_fem_stage.py (Line 47)
    _plot_fem_vs_exact_internal(xpoints, (rho_ex, u_ex, p_ex, e_ex), (rho_fem, u_fem, p_fem, e_fem), plot_path)

def _plot_fem_vs_exact_internal(
    x: np.ndarray, 
    exact: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fem: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    save_path: str
) -> None:
    """
    Plots the 2x2 grid comparing the exact Riemann solution to the SUPG-YZβ FEM solution.
    Expects tuples of (rho, u, p, e).
    """
    rho_ex, u_ex, p_ex, e_ex = exact
    rho_fem, u_fem, p_fem, e_fem = fem

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    panels = [
        (ax[0,0], rho_fem, rho_ex, r'$\rho$', 'Density'),
        (ax[0,1], u_fem,   u_ex,   'u',       'Velocity'),
        (ax[1,0], p_fem,   p_ex,   'p',       'Pressure'),
        (ax[1,1], e_fem,   e_ex,   'e',       'Specific internal energy')
    ]

    for axi, num, exa, ylab, title in panels:
        axi.plot(x, exa, 'k-', lw=1.5, label='Exact')
        axi.plot(x, num, 'r-', lw=0.8, label=r'SUPG-YZ$\beta$')
        axi.set_xlabel('x')
        axi.set_ylabel(ylab)
        axi.set_title(title)
        axi.legend()
        axi.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
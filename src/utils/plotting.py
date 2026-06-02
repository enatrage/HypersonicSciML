import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Tuple

def plot_fem_vs_exact(
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


def plot_pinn_vs_fem_vs_exact(
    x: np.ndarray, 
    exact: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    fem: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    pinn: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
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
    plt.close(fig)


def plot_training_diagnostics(history: Dict[str, List[float]], save_path: str) -> None:
    """
    Plots the PDE/Data loss curves and the adaptive weighting schedule over time.
    """
    if not history or not history.get("epoch"):
        print("[INFO] No history provided to plotter. Skipping diagnostics.")
        return

    fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
    ep = history["epoch"]
    
    ax2[0].semilogy(ep, history["L_total"], 'b-',  label='Total', lw=1)
    ax2[0].semilogy(ep, history["L_data"],  'g--', label='Data',  lw=1)
    ax2[0].semilogy(ep, [max(v, 1e-12) for v in history["L_pde"]], 'r:', label='PDE', lw=1)
    ax2[0].set_xlabel('epoch')
    ax2[0].set_ylabel('loss')
    ax2[0].set_title('Loss components')
    ax2[0].grid(alpha=0.3)
    ax2[0].legend()

    ax2[1].plot(ep, history["w_data"], 'g-',  label=r'$w_{data}$', lw=1.5)
    ax2[1].plot(ep, history["w_pde"],  'r--', label=r'$w_{pde}$',  lw=1.5)
    ax2[1].set_xlabel('epoch')
    ax2[1].set_ylabel('weight')
    ax2[1].set_title('Adaptive weight schedule')
    ax2[1].grid(alpha=0.3)
    ax2[1].legend()

    fig2.tight_layout()
    fig2.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig2)
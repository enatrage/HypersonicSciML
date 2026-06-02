import numpy as np
from dataclasses import dataclass

@dataclass
class FemSnapshots:
    """In-memory view of the .npz file produced by run_fem()."""
    x: np.ndarray
    t: np.ndarray
    U: np.ndarray             # (K_s, Nx, 3) = (rho, q, E)
    gamma: float
    U_ref: np.ndarray
    t_final: float
    dt: float
    fingerprint: str = ""
    
    @property
    def K_s(self): return self.t.size
    
    @property
    def Nx(self):  return self.x.size
    
    @classmethod
    def load(cls, path: str) -> "FemSnapshots":
        d = np.load(path)
        U = np.stack([d["rho_snap"], d["q_snap"], d["E_snap"]], axis=-1)
        fp = str(d["fem_fingerprint"]) if "fem_fingerprint" in d.files else ""
        return cls(x       = d["x"].astype(np.float64),
                   t       = d["t_snap"].astype(np.float64),
                   U       = U.astype(np.float64),
                   gamma   = float(d["gamma"]),
                   U_ref   = d["U_ref"].astype(np.float64),
                   t_final = float(d["t_final"]),
                   dt      = float(d["dt"]),
                   fingerprint = fp)

def smoothness_mask(fem: FemSnapshots, t_q: np.ndarray, x_q: np.ndarray, grad_quantile: float) -> np.ndarray:
    """
    Boolean mask of collocation points that lie in smooth regions of the FEM solution.
    """
    dx_fem  = fem.x[1] - fem.x[0]
    drho_dx = np.gradient(fem.U[..., 0], dx_fem, axis=1)
    abs_grad = np.abs(drho_dx)
    thresh  = np.quantile(abs_grad[-1], grad_quantile)
    bad     = abs_grad > thresh
    t_idx   = np.argmin(np.abs(fem.t[None, :] - t_q[:, None]), axis=1)
    x_idx   = np.clip(np.searchsorted(fem.x, x_q), 0, fem.Nx - 1)
    return ~bad[t_idx, x_idx]
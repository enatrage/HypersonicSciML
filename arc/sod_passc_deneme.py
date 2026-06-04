#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sod_passc.py
============

PASSC-Transient for the 1-D Sod shock tube --- ALL IN ONE FILE.

Pipeline (executed in order):

  STEP 1  FEM       : P1 Galerkin + system SUPG + YZβ shock-capturing,
                      backward-Euler time stepping in FEniCS.  Records
                      the last K_s temporal snapshots of U = (ρ, ρu, E).
  STEP 2  PINN      : Random Fourier features -> n_r residual blocks ->
                      narrowing -> 3-output head, trained in PyTorch on
                      the FEM snapshots with selective PDE-residual
                      enforcement in smooth regions (gradient-mask).
  STEP 3  Compare   : Both solutions evaluated at t = T against the
                      exact Riemann solution (Toro, 2009).
  STEP 4  Plot      : 4-panel comparison + training diagnostics.

Run as a script:  python3 sod_passc.py

Caching
-------
Both steps cache their results and re-run only when their inputs change:

  FEM cache  : `sod_fem_snapshots.npz` carries a "fem_fingerprint" string
               derived from GAMMA_V, RHO_L_V, ..., NX, DT, T_FINAL, BETA_V,
               K_S.  If the file exists and its fingerprint matches the
               current settings, the FEM step is skipped automatically.

  PINN cache : `sod_passc_model.pt` carries a "pinn_fingerprint" derived
               from the FEM fingerprint + every TrainConfig field.  If
               that matches, the network is loaded and evaluation runs
               without retraining.

Force flags
-----------
  --force-fem    : recompute FEM even if cache is valid.
  --force-pinn   : retrain PINN even if cache is valid.
  --skip-fem     : never run FEM; fail if no usable snapshot exists.
  --skip-pinn    : never run PINN.

All scalar PINN hyperparameters are defined ONCE in the `TrainConfig`
dataclass; the argparse CLI is auto-generated from it.

Dependencies
------------
  FEM  step  : FEniCS (2019.x), mshr, numpy, scipy, matplotlib
  PINN step  : PyTorch, numpy, scipy, matplotlib

@author: scengizci  (FEM)  +  Claude (PASSC post-processor)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from typing import Optional
import time


# ===========================================================================
#                       PROBLEM CONSTANTS  (used by both halves)
# ===========================================================================
GAMMA_V        = 1.4
RHO_L_V        = 1.0
U_L_V          = 0.0
P_L_V          = 1.0
RHO_R_V        = 0.125
U_R_V          = 0.0
P_R_V          = 0.1
X_DIAPH        = 0.5

T_FINAL        = 0.20
DT             = 1.0e-3
NX             = 250
BETA_V         = 2.0
K_S            = 10                                  # snapshots to retain

SNAPSHOT_FILE  = "runs/fem/refactor/older/sod_fem_snapshots.npz"
PINN_CACHE     = "runs/pinn/refactor/older/sod_passc_model.pt"
PASSC_PLOT     = "runs/pinn/refactor/older/sod_passc.png"
FEM_PLOT       = "runs/fem/refactor/older/sod_cyz2.png"

# Reference scales used by YZβ and by the PINN's Y-scaled losses.
U1_REF_V = float(RHO_L_V)
U2_REF_V = float(RHO_L_V * 1.0)                                      # ~ρ_L * a_L
U3_REF_V = float(P_L_V / (GAMMA_V - 1.0) + 0.5 * RHO_L_V * 1.0)      # ~E_L


# ===========================================================================
#                            Fingerprint helpers
#
#  A "fingerprint" is a short hex string deterministically derived from a
#  dict of inputs.  Two runs with the same inputs produce the same string;
#  any change produces a different one.  This is how cache files know
#  whether they are still valid.
# ===========================================================================

def _fingerprint(d: dict) -> str:
    """SHA-1 hex digest (first 16 chars) of a JSON-canonicalised dict."""
    blob = json.dumps(d, sort_keys=True, separators=(",", ":"),
                      default=str).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def fem_inputs() -> dict:
    """The set of constants the FEM step depends on."""
    return dict(
        GAMMA_V=GAMMA_V,
        RHO_L_V=RHO_L_V, U_L_V=U_L_V, P_L_V=P_L_V,
        RHO_R_V=RHO_R_V, U_R_V=U_R_V, P_R_V=P_R_V,
        X_DIAPH=X_DIAPH,
        T_FINAL=T_FINAL, DT=DT, NX=NX, BETA_V=BETA_V, K_S=K_S,
        U1_REF_V=U1_REF_V, U2_REF_V=U2_REF_V, U3_REF_V=U3_REF_V,
    )


def fem_fingerprint() -> str:
    return _fingerprint(fem_inputs())


# ===========================================================================
#                                STEP 1 : FEM
#       (P1 + system SUPG + YZβ, identical to the original sod_fem.py)
# ===========================================================================

def run_fem(snapshot_path: str = SNAPSHOT_FILE) -> None:
    """
    Run the SUPG-YZβ Sod-shock-tube solve in FEniCS and write the snapshot
    bundle `<snapshot_path>` for the PINN step.  Also produces `sod_cyz2.png`
    (exact vs FEM comparison), matching the original `sod_fem.py` output.
    """
    # ----- numpy compat shims (some FFC/UFL versions still call these) -----
    import numpy as _np
    if not hasattr(_np, "product"):       _np.product       = _np.prod
    if not hasattr(_np, "cumproduct"):    _np.cumproduct    = _np.cumprod
    if not hasattr(_np, "alltrue"):       _np.alltrue       = _np.all
    if not hasattr(_np, "sometrue"):      _np.sometrue      = _np.any

    from fenics import (
        Constant, IntervalMesh, FiniteElement, MixedElement, FunctionSpace,
        Function, TestFunctions, split, as_vector, as_matrix, dot, sqrt, dx,
        CellDiameter, UserExpression, interpolate, derivative,
        NonlinearVariationalProblem, NonlinearVariationalSolver,
        File, parameters, set_log_level, LogLevel,
    )

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import brentq
    sys.setrecursionlimit(1000000000)

    parameters["allow_extrapolation"] = True
    parameters["form_compiler"]["optimize"] = True
    parameters["form_compiler"]["representation"] = "uflacs"
    parameters["form_compiler"]["no-evaluate_basis_derivatives"] = False
    parameters['form_compiler']['quadrature_degree'] = 9
    set_log_level(LogLevel.INFO)

    # ----- problem constants (FEniCS Constants for the form) -----
    gamma   = Constant(GAMMA_V)
    gamma_v = GAMMA_V
    rho_L_v, u_L_v, p_L_v = RHO_L_V, U_L_V, P_L_V
    rho_R_v, u_R_v, p_R_v = RHO_R_V, U_R_V, P_R_V
    x_diaph = X_DIAPH
    U1_ref  = Constant(U1_REF_V)
    U2_ref  = Constant(U2_REF_V)
    U3_ref  = Constant(U3_REF_V)
    beta_v  = BETA_V

    # ----- mesh & function space -----
    mesh = IntervalMesh(NX, 0.0, 1.0)
    print("Number of Cells:", mesh.num_cells())
    print("Number of Nodes:", mesh.num_vertices())
    from fenics import interval as _interval
    P1 = FiniteElement('P', _interval, 1)
    element = MixedElement([P1, P1, P1])
    V = FunctionSpace(mesh, element)

    # ----- solution functions -----
    U   = Function(V)
    rho, q, E = split(U)
    U_n = Function(V)
    rho_n, q_n, E_n = split(U_n)
    phi1, phi2, phi3 = TestFunctions(V)
    U_vec   = as_vector([rho,   q,   E])
    U_n_vec = as_vector([rho_n, q_n, E_n])
    W_vec   = as_vector([phi1,  phi2, phi3])

    # ----- initial conditions -----
    class InitialConditions(UserExpression):
        def eval(self, values, x):
            if x[0] < x_diaph:
                r, u, p = rho_L_v, u_L_v, p_L_v
            else:
                r, u, p = rho_R_v, u_R_v, p_R_v
            values[0] = r
            values[1] = r * u
            values[2] = p / (gamma_v - 1.0) + 0.5 * r * u * u
        def value_shape(self):
            return (3,)
    init = InitialConditions(degree=2)
    U_n  = interpolate(init, V)
    U.assign(U_n)
    rho_n, q_n, E_n = split(U_n)
    U_n_vec = as_vector([rho_n, q_n, E_n])

    # ----- primitive variables, current step -----
    u_expr = q / rho
    p_expr = (gamma - 1.0) * (E - 0.5 * q * q / rho)
    H_expr = (E + p_expr) / rho
    a_expr = sqrt(gamma * p_expr / rho)
    # ----- primitive variables, previous step (used by τ and A_lin) -----
    u_n_expr = q_n / rho_n
    p_n_expr = (gamma - 1.0) * (E_n - 0.5 * q_n * q_n / rho_n)
    H_n_expr = (E_n + p_n_expr) / rho_n
    a_n_expr = sqrt(gamma * p_n_expr / rho_n)

    # ----- flux Jacobian A = dF/dU -----
    def jacobian_A(rho_, q_, E_):
        u_  = q_ / rho_
        p_  = (gamma - 1.0) * (E_ - 0.5 * q_ * q_ / rho_)
        H_  = (E_ + p_) / rho_
        A11 = Constant(0.0); A12 = Constant(1.0); A13 = Constant(0.0)
        A21 = 0.5 * (gamma - 3.0) * u_ * u_
        A22 = (3.0 - gamma) * u_
        A23 = gamma - 1.0
        A31 = u_ * (0.5 * (gamma - 1.0) * u_ * u_ - H_)
        A32 = H_ - (gamma - 1.0) * u_ * u_
        A33 = gamma * u_
        return as_matrix([[A11, A12, A13],
                          [A21, A22, A23],
                          [A31, A32, A33]])
    A_cur = jacobian_A(rho,   q,   E)
    A_lin = jacobian_A(rho_n, q_n, E_n)

    dU_dx   = as_vector([rho.dx(0),   q.dx(0),   E.dx(0)])
    dU_n_dx = as_vector([rho_n.dx(0), q_n.dx(0), E_n.dx(0)])
    dW_dx   = as_vector([phi1.dx(0),  phi2.dx(0), phi3.dx(0)])

    # ----- strong residual & stabilisation -----
    R_strong = (U_vec - U_n_vec) / DT + A_cur * dU_dx
    h        = CellDiameter(mesh)
    smax     = abs(u_n_expr) + a_n_expr
    tau_SUPG = h / (2.0 * smax)

    # ----- YZβ shock-capturing (CYZ, β=2) -----
    Yinv_vec = as_vector([1.0/U1_ref, 1.0/U2_ref, 1.0/U3_ref])
    Z_vec    = A_cur * dU_dx
    YinvZ    = as_vector([Yinv_vec[0]*Z_vec[0],
                          Yinv_vec[1]*Z_vec[1],
                          Yinv_vec[2]*Z_vec[2]])
    YinvdU   = as_vector([Yinv_vec[0]*dU_dx[0],
                          Yinv_vec[1]*dU_dx[1],
                          Yinv_vec[2]*dU_dx[2]])
    eps_floor    = Constant(1.0e-12)
    normYinvZ_sq = YinvZ[0]**2  + YinvZ[1]**2  + YinvZ[2]**2  + eps_floor
    normYinvdU_2 = YinvdU[0]**2 + YinvdU[1]**2 + YinvdU[2]**2 + eps_floor
    h_SHOC  = h
    nu_SHOC = (sqrt(normYinvZ_sq)
               * normYinvdU_2**(beta_v/2.0 - 1.0)
               * (h_SHOC/2.0)**beta_v)

    # ----- weak form -----
    GAL  = dot(W_vec, (U_vec - U_n_vec)/DT) * dx + dot(W_vec, A_cur * dU_dx) * dx
    A_lin_T = as_matrix([[A_lin[0,0], A_lin[1,0], A_lin[2,0]],
                         [A_lin[0,1], A_lin[1,1], A_lin[2,1]],
                         [A_lin[0,2], A_lin[1,2], A_lin[2,2]]])
    AT_dW = A_lin_T * dW_dx
    SUPG  = tau_SUPG * dot(AT_dW, R_strong) * dx
    SHOC  = nu_SHOC * dot(dW_dx, dU_dx) * dx
    F     = GAL + SUPG + SHOC

    problem = NonlinearVariationalProblem(F, U, J=derivative(F, U))
    solver  = NonlinearVariationalSolver(problem)
    prm = solver.parameters["newton_solver"]
    prm["absolute_tolerance"] = 1E-10
    prm['relative_tolerance'] = 1E-10
    prm['maximum_iterations'] = 50
    prm['convergence_criterion'] = 'incremental'
    prm['krylov_solver']['absolute_tolerance']      = 1E-10
    prm['krylov_solver']['relative_tolerance']      = 1E-10
    prm['krylov_solver']['maximum_iterations']      = 1000
    prm['krylov_solver']['monitor_convergence']     = True
    prm['krylov_solver']['nonzero_initial_guess']   = True
    prm['krylov_solver']['error_on_nonconvergence'] = True
    prm['krylov_solver']['report']                  = True

    file_rho = File("results/rho.pvd")
    file_q   = File("results/q.pvd")
    file_E   = File("results/E.pvd")

    # ---- snapshot bookkeeping ----
    x_nodes        = mesh.coordinates().flatten().astype(np.float64)
    x_nodes_sorted = x_nodes[np.argsort(x_nodes)]
    snap_times, snap_rho, snap_q, snap_E = [], [], [], []
    def _nodal(field):
        out = np.empty_like(x_nodes_sorted)
        for i, xp in enumerate(x_nodes_sorted):
            out[i] = field(xp)
        return out

    # ---- time loop ----
    t = 0.0
    while t < T_FINAL - 1e-14:
        dt_local = (T_FINAL - t) if (t + DT > T_FINAL) else DT
        t += dt_local
        solver.solve()
        rho_sol, q_sol, E_sol = U.split()
        file_rho << (rho_sol, t)
        file_q   << (q_sol,   t)
        file_E   << (E_sol,   t)
        snap_times.append(float(t))
        snap_rho.append(_nodal(rho_sol))
        snap_q  .append(_nodal(q_sol))
        snap_E  .append(_nodal(E_sol))
        if len(snap_times) > K_S:
            snap_times.pop(0); snap_rho.pop(0); snap_q.pop(0); snap_E.pop(0)
        U_n.assign(U)
        rho_n, q_n, E_n = split(U_n)
        U_n_vec = as_vector([rho_n, q_n, E_n])
        print("t =", t)

    # ---- write the snapshot bundle (now with fingerprint) ----
    fp = fem_fingerprint()
    np.savez(snapshot_path,
             x        = x_nodes_sorted,
             t_snap   = np.asarray(snap_times,             dtype=np.float64),
             rho_snap = np.stack(snap_rho, axis=0).astype(np.float64),
             q_snap   = np.stack(snap_q,   axis=0).astype(np.float64),
             E_snap   = np.stack(snap_E,   axis=0).astype(np.float64),
             gamma    = np.float64(GAMMA_V),
             U_ref    = np.array([U1_REF_V, U2_REF_V, U3_REF_V], dtype=np.float64),
             t_final  = np.float64(snap_times[-1]),
             dt       = np.float64(DT),
             nx       = np.int64(NX),
             fem_fingerprint = np.array(fp))
    print(f"  -> {snapshot_path} saved ({K_S} snapshots, "
          f"{x_nodes_sorted.size} nodes each, fingerprint={fp})")

    # ---- side-by-side plot vs the exact Riemann solution (FEM only) ----
    rho_sol, q_sol, E_sol = U.split(deepcopy=True)
    xpoints = np.linspace(0.0, 1.0, 1000)
    rho_num = np.array([rho_sol(xp) for xp in xpoints])
    q_num   = np.array([q_sol(xp)   for xp in xpoints])
    E_num   = np.array([E_sol(xp)   for xp in xpoints])
    u_num   = q_num / rho_num
    p_num   = (GAMMA_V - 1.0) * (E_num - 0.5 * rho_num * u_num**2)
    e_num   = p_num / ((GAMMA_V - 1.0) * rho_num)

    rho_ex, u_ex, p_ex = exact_riemann(xpoints, T_FINAL)
    e_ex = p_ex / ((GAMMA_V - 1.0) * rho_ex)

    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    for axi, num, exa, lab in [(ax[0,0], rho_num, rho_ex, r'$\rho$'),
                               (ax[0,1], u_num,   u_ex,   'u'),
                               (ax[1,0], p_num,   p_ex,   'p'),
                               (ax[1,1], e_num,   e_ex,   'e')]:
        axi.plot(xpoints, exa, 'k-', lw=1.5, label='Exact')
        axi.plot(xpoints, num, 'r-', lw=0.8, label=r'SUPG-YZ$\beta$')
        axi.set_xlabel('x'); axi.set_ylabel(lab)
        axi.legend(); axi.grid(alpha=0.3)
    ax[0,0].set_title('Density')
    ax[0,1].set_title('Velocity')
    ax[1,0].set_title('Pressure')
    ax[1,1].set_title('Specific internal energy')
    fig.tight_layout()
    fig.savefig(FEM_PLOT, dpi=600, bbox_inches='tight')
    plt.close(fig)
    print(f"  -> {FEM_PLOT} saved")


def fem_cache_is_valid(snapshot_path: str) -> bool:
    """
    True if `snapshot_path` exists and its embedded `fem_fingerprint`
    matches the current problem constants.  Old snapshots produced before
    fingerprinting was added are treated as invalid (safer to recompute).
    """
    import numpy as np
    if not os.path.exists(snapshot_path):
        return False
    try:
        with np.load(snapshot_path, allow_pickle=False) as d:
            if "fem_fingerprint" not in d.files:
                return False
            stored = str(d["fem_fingerprint"])
    except Exception:
        return False
    return stored == fem_fingerprint()


# ===========================================================================
#                       Exact Riemann solver (Toro, 2009)
# ===========================================================================

def exact_riemann(x_arr, t_eval,
                  gamma_v=GAMMA_V,
                  rho_L=RHO_L_V, u_L=U_L_V, p_L=P_L_V,
                  rho_R=RHO_R_V, u_R=U_R_V, p_R=P_R_V,
                  x0=X_DIAPH):
    import numpy as np
    from scipy.optimize import brentq
    g  = gamma_v
    aL = np.sqrt(g * p_L / rho_L)
    aR = np.sqrt(g * p_R / rho_R)
    def fK(p, rhoK, pK, aK):
        if p > pK:
            AK = 2.0/((g+1.0)*rhoK); BK = (g-1.0)/(g+1.0)*pK
            return (p - pK) * np.sqrt(AK/(p + BK))
        return 2.0*aK/(g-1.0) * ((p/pK)**((g-1.0)/(2.0*g)) - 1.0)
    Ffun  = lambda p: fK(p,rho_L,p_L,aL) + fK(p,rho_R,p_R,aR) + (u_R - u_L)
    p_star = brentq(Ffun, 1e-8, 10.0*max(p_L,p_R))
    u_star = 0.5*(u_L+u_R) + 0.5*(fK(p_star,rho_R,p_R,aR)
                                  - fK(p_star,rho_L,p_L,aL))
    rho_=np.zeros_like(x_arr); u_=np.zeros_like(x_arr); p_=np.zeros_like(x_arr)
    for i, xi in enumerate(x_arr):
        s = (xi - x0) / t_eval
        if s <= u_star:
            if p_star <= p_L:
                rsL = rho_L*(p_star/p_L)**(1.0/g)
                asL = aL*(p_star/p_L)**((g-1.0)/(2.0*g))
                SHL = u_L - aL; STL = u_star - asL
                if s < SHL:    rho_[i],u_[i],p_[i] = rho_L, u_L, p_L
                elif s > STL:  rho_[i],u_[i],p_[i] = rsL, u_star, p_star
                else:
                    rho_[i] = rho_L*(2.0/(g+1.0)+(g-1.0)/((g+1.0)*aL)*(u_L-s))**(2.0/(g-1.0))
                    u_[i]   = 2.0/(g+1.0)*(aL+(g-1.0)*0.5*u_L+s)
                    p_[i]   = p_L*(2.0/(g+1.0)+(g-1.0)/((g+1.0)*aL)*(u_L-s))**(2.0*g/(g-1.0))
            else:
                pr = p_star/p_L
                rsL = rho_L*(pr+(g-1.0)/(g+1.0))/((g-1.0)/(g+1.0)*pr+1.0)
                SL  = u_L-aL*np.sqrt((g+1.0)/(2.0*g)*pr+(g-1.0)/(2.0*g))
                if s < SL:     rho_[i],u_[i],p_[i] = rho_L, u_L, p_L
                else:          rho_[i],u_[i],p_[i] = rsL, u_star, p_star
        else:
            if p_star <= p_R:
                rsR = rho_R*(p_star/p_R)**(1.0/g)
                asR = aR*(p_star/p_R)**((g-1.0)/(2.0*g))
                SHR = u_R + aR; STR = u_star + asR
                if s > SHR:    rho_[i],u_[i],p_[i] = rho_R, u_R, p_R
                elif s < STR:  rho_[i],u_[i],p_[i] = rsR, u_star, p_star
                else:
                    rho_[i] = rho_R*(2.0/(g+1.0)-(g-1.0)/((g+1.0)*aR)*(u_R-s))**(2.0/(g-1.0))
                    u_[i]   = 2.0/(g+1.0)*(-aR+(g-1.0)*0.5*u_R+s)
                    p_[i]   = p_R*(2.0/(g+1.0)-(g-1.0)/((g+1.0)*aR)*(u_R-s))**(2.0*g/(g-1.0))
            else:
                pr = p_star/p_R
                rsR = rho_R*(pr+(g-1.0)/(g+1.0))/((g-1.0)/(g+1.0)*pr+1.0)
                SR  = u_R+aR*np.sqrt((g+1.0)/(2.0*g)*pr+(g-1.0)/(2.0*g))
                if s > SR:     rho_[i],u_[i],p_[i] = rho_R, u_R, p_R
                else:          rho_[i],u_[i],p_[i] = rsR, u_star, p_star
    return rho_, u_, p_


# ===========================================================================
#                              STEP 2 : PINN
# ===========================================================================

@dataclass
class FemSnapshots:
    """In-memory view of the .npz file produced by run_fem()."""
    x: "np.ndarray"
    t: "np.ndarray"
    U: "np.ndarray"             # (K_s, Nx, 3) = (rho, q, E)
    gamma: float
    U_ref: "np.ndarray"
    t_final: float
    dt: float
    fingerprint: str = ""
    @property
    def K_s(self): return self.t.size
    @property
    def Nx(self):  return self.x.size
    @classmethod
    def load(cls, path: str) -> "FemSnapshots":
        import numpy as np
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


@dataclass
class Phase:
    upto_epoch: int
    w_data:     float
    w_pde:      float


# ---------------------------------------------------------------------------
#                       SINGLE SOURCE OF TRUTH
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    # Network capacity
    n_hidden:    int   = field(default=48,
                               metadata={"help": "Hidden layer width."})
    n_blocks:    int   = field(default=6,
                               metadata={"help": "Number of residual blocks."})
    n_fourier:   int   = field(default=24,
                               metadata={"help": "Fourier feature count."})
    sigma:       float = field(default=4.0,
                               metadata={"help": "Fourier feature scale."})
    # Optimisation
    epochs:      int   = field(default=5000,
                               metadata={"help": "Training epochs."})
    batch_size:  int   = field(default=1024,
                               metadata={"help": "Mini-batch size."})
    lr:          float = field(default=8.0e-4,
                               metadata={"help": "AdamW learning rate."})
    weight_decay:float = field(default=1.0e-7,
                               metadata={"help": "AdamW weight decay."})
    n_coll:      int   = field(default=2048,
                               metadata={"help": "Collocation points per PDE step."})
    grad_clip:   float = field(default=1.0,
                               metadata={"help": "Gradient-norm clip."})
    res_clip:    float = field(default=10.0,
                               metadata={"help": "PDE residual clip value."})
    # Selective enforcement -- top (1-q) fraction of |drho/dx| nodes are excluded
    grad_quantile:       float = field(default=0.92,
                                       metadata={"help": "Smoothness quantile threshold."})
    pde_every_k_batches: int   = field(default=1,
                                       metadata={"help": "Evaluate PDE loss every k batches."})
    seed:        int   = field(default=0,
                               metadata={"help": "RNG seed."})
    # Multi-phase weight schedule (not exposed via CLI -- edit in code)
    schedule: list = field(default_factory=lambda: [
        Phase(upto_epoch=1000, w_data=1.00, w_pde=0.1),
        Phase(upto_epoch=2000, w_data=0.60, w_pde=0.60),
        Phase(upto_epoch=3500, w_data=0.30, w_pde=5.00),
        Phase(upto_epoch=10**9, w_data=0.01, w_pde=15.00),   # physics-dominant
    ])


def trainconfig_to_dict(cfg: TrainConfig) -> dict:
    """Stable serialisation of TrainConfig including the schedule."""
    d = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name == "schedule":
            d["schedule"] = [asdict(ph) for ph in v]
        else:
            d[f.name] = v
    return d


def pinn_fingerprint(cfg: TrainConfig, fem_fp: str) -> str:
    payload = {"fem_fingerprint": fem_fp,
               "train_config":    trainconfig_to_dict(cfg)}
    return _fingerprint(payload)


def _build_pinn_modules():
    """
    Imports PyTorch lazily and returns the (FourierEmbedding, ResidualBlock,
    PINN_Euler) classes plus the autograd helper.
    """
    import torch
    import torch.nn as nn

    class FourierEmbedding(nn.Module):
        def __init__(self, input_dim, n_features, sigma):
            super().__init__()
            B = torch.randn(input_dim, n_features) * sigma
            self.register_buffer("B", B)
            self.out_dim = input_dim + 2 * n_features
        def forward(self, z):
            proj = z @ self.B
            return torch.cat([z, torch.sin(proj), torch.cos(proj)], dim=-1)

    class ResidualBlock(nn.Module):
        def __init__(self, n_h):
            super().__init__()
            self.lin1 = nn.Linear(n_h, n_h)
            self.lin2 = nn.Linear(n_h, n_h)
            self.act  = nn.SiLU()
            self.norm = nn.LayerNorm(n_h)
        def forward(self, h):
            p = self.act(self.lin1(h))
            q = self.lin2(p)
            return self.act(self.norm(q + h))

    class PINN_Euler(nn.Module):
        """
        u_NN(t, x) = (rho, q, E)
        rho and E are kept strictly positive via softplus reparametrisation
        so the PDE residual never sees a negative density or energy.
        """
        def __init__(self, n_hidden=48, n_blocks=6, n_fourier=24, sigma=4.0,
                     rho_floor=1.0e-3, E_floor=1.0e-3):
            super().__init__()
            self.embed = FourierEmbedding(2, n_fourier, sigma)
            self.input_layer = nn.Linear(self.embed.out_dim, n_hidden)
            self.blocks = nn.ModuleList(
                [ResidualBlock(n_hidden) for _ in range(n_blocks)])
            self.narrow = nn.Linear(n_hidden, n_hidden // 2)
            self.head   = nn.Linear(n_hidden // 2, 3)
            self.act    = nn.SiLU()
            self.rho_floor = rho_floor
            self.E_floor   = E_floor
        def forward(self, t, x):
            z = torch.cat([t, x], dim=-1)
            h = self.act(self.input_layer(self.embed(z)))
            for blk in self.blocks:
                h = blk(h)
            h = self.act(self.narrow(h))
            raw = self.head(h)
            rho_raw, q_raw, E_raw = raw[:, 0:1], raw[:, 1:2], raw[:, 2:3]
            rho = self.rho_floor + nn.functional.softplus(rho_raw)
            E   = self.E_floor   + nn.functional.softplus(E_raw)
            return torch.cat([rho, q_raw, E], dim=-1)

    def euler_fluxes(U, gamma):
        rho = U[:, 0:1]; q = U[:, 1:2]; E = U[:, 2:3]
        rho_safe = rho.clamp(min=1.0e-8)
        u = q / rho_safe
        p = (gamma - 1.0) * (E - 0.5 * q * u)
        return torch.cat([q, q * u + p, (E + p) * u], dim=-1)

    def pde_residual(model, t, x, gamma):
        U = model(t, x)
        F = euler_fluxes(U, gamma)
        Ut_cols, Fx_cols = [], []
        for k in range(3):
            Uk = U[:, k:k+1]; Fk = F[:, k:k+1]
            Ut_cols.append(torch.autograd.grad(
                Uk, t, grad_outputs=torch.ones_like(Uk),
                create_graph=True, retain_graph=True)[0])
            Fx_cols.append(torch.autograd.grad(
                Fk, x, grad_outputs=torch.ones_like(Fk),
                create_graph=True, retain_graph=True)[0])
        return torch.cat(Ut_cols, dim=-1) + torch.cat(Fx_cols, dim=-1)

    return FourierEmbedding, ResidualBlock, PINN_Euler, pde_residual


def _smoothness_mask(fem: FemSnapshots, t_q, x_q, grad_quantile):
    """
    Boolean mask of collocation points that lie in **smooth** regions of the
    FEM solution.  We declare a point "rough" if its nearest-neighbour FEM
    node has |drho/dx| above the global gradient quantile.
    """
    import numpy as np
    dx_fem  = fem.x[1] - fem.x[0]
    drho_dx = np.gradient(fem.U[..., 0], dx_fem, axis=1)
    abs_grad = np.abs(drho_dx)
    thresh  = np.quantile(abs_grad[-1], grad_quantile)
    bad     = abs_grad > thresh
    t_idx   = np.argmin(np.abs(fem.t[None, :] - t_q[:, None]), axis=1)
    x_idx   = np.clip(np.searchsorted(fem.x, x_q), 0, fem.Nx - 1)
    return ~bad[t_idx, x_idx]


def _build_model_from_cfg(cfg: TrainConfig, device):
    """Instantiate a PINN_Euler with the given config, on the given device."""
    _, _, PINN_Euler, _ = _build_pinn_modules()
    model = PINN_Euler(
        n_hidden=cfg.n_hidden, n_blocks=cfg.n_blocks,
        n_fourier=cfg.n_fourier, sigma=cfg.sigma,
    ).to(device)
    return model


def pinn_cache_is_valid(cache_path: str, target_fp: str) -> bool:
    """Quick check: does `cache_path` exist with a matching fingerprint?"""
    if not os.path.exists(cache_path):
        return False
    try:
        import torch
        # weights_only=False because we also store the config dict + history.
        # The file is produced by this same script, so it is trusted local data.
        ckpt = torch.load(cache_path, map_location="cpu", weights_only=False)
        return ckpt.get("pinn_fingerprint", "") == target_fp
    except Exception:
        return False


def run_pinn(snapshot_path: str = SNAPSHOT_FILE,
             cfg: Optional[TrainConfig] = None,
             device_str: str = "auto",
             out_plot: str = PASSC_PLOT,
             cache_path: str = PINN_CACHE,
             force_retrain: bool = False) -> dict:
    """
    Train (or load from cache) the PASSC-Transient PINN on FEM snapshots
    stored at `snapshot_path`, and write a 4-panel comparison plot + training
    diagnostics.  Returns the L2 error summary against the exact Riemann
    solution at t = T.
    """
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(
            f"snapshot file not found: {snapshot_path}\n"
            f"Run the FEM step first (sod_passc.py without --skip-fem).")

    if cfg is None:
        cfg = TrainConfig()

    # ----- deterministic seeds -----
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[INFO] PINN device = {device}")

    fem = FemSnapshots.load(snapshot_path)
    print(f"[INFO] {fem.K_s} snapshots, {fem.Nx} nodes, "
          f"t in [{fem.t[0]:.4f}, {fem.t[-1]:.4f}], gamma = {fem.gamma}")
    print(f"[INFO] U_ref = {fem.U_ref}")
    print(f"[INFO] FEM fingerprint = {fem.fingerprint or '<none>'}")

    target_fp = pinn_fingerprint(cfg, fem.fingerprint)
    print(f"[INFO] PINN fingerprint = {target_fp}")

    # ----- prepare evaluation tensors (used in both cached and trained paths) -----
    U_ref = torch.from_numpy(fem.U_ref).float().to(device)
    Y_inv = (1.0 / U_ref).view(1, 3)

    # ----- try cache -----
    use_cache = (not force_retrain) and pinn_cache_is_valid(cache_path, target_fp)
    history: dict = {}
    if use_cache:
        print(f"[INFO] PINN cache hit: loading {cache_path}")
        ckpt = torch.load(cache_path, map_location=device, weights_only=False)
        model = _build_model_from_cfg(cfg, device)
        model.load_state_dict(ckpt["model_state"])
        history = ckpt.get("history", {})
    else:
        if force_retrain:
            print("[INFO] --force-pinn set: retraining from scratch.")
        elif os.path.exists(cache_path):
            print("[INFO] PINN cache present but fingerprint differs; retraining.")
        else:
            print(f"[INFO] No PINN cache at {cache_path}; training.")

        _, _, PINN_Euler, pde_residual = _build_pinn_modules()

        # ----- flatten training data -----
        K_s, Nx = fem.K_s, fem.Nx
        tt = np.broadcast_to(fem.t[:, None], (K_s, Nx)).reshape(-1, 1)
        xx = np.broadcast_to(fem.x[None, :], (K_s, Nx)).reshape(-1, 1)
        UU = fem.U.reshape(-1, 3)
        tt = torch.from_numpy(tt).float().to(device)
        xx = torch.from_numpy(xx).float().to(device)
        UU = torch.from_numpy(UU).float().to(device)

        t_start = float(fem.t.min()); t_end = float(fem.t.max())
        x_min   = float(fem.x.min()); x_max = float(fem.x.max())

        # ----- model & optimiser -----
        model = _build_model_from_cfg(cfg, device)
        optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                  weight_decay=cfg.weight_decay,
                                  betas=(0.9, 0.999))
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optim, mode='min', factor=0.9, patience=150, min_lr=1.0e-6)

        history = {k: [] for k in ("epoch","L_total","L_data","L_pde",
                                   "w_data","w_pde","lr")}
        n_data = tt.shape[0]
        steps_per_epoch = max(1, math.ceil(n_data / cfg.batch_size))
        best_state, best_loss = None, float("inf")

        def _weights(epoch):
            for ph in cfg.schedule:
                if epoch < ph.upto_epoch: return ph.w_data, ph.w_pde
            return cfg.schedule[-1].w_data, cfg.schedule[-1].w_pde


        start_time = time.time()
        # ----- training loop -----
        log_every = max(1, cfg.epochs // 250)
        for epoch in range(cfg.epochs):
            w_data, w_pde = _weights(epoch)
            perm = torch.randperm(n_data, device=device)
            ep_total = ep_data = ep_pde = 0.0
            for step in range(steps_per_epoch):
                sel = perm[step*cfg.batch_size:(step+1)*cfg.batch_size]
                t_b, x_b, U_b = tt[sel], xx[sel], UU[sel]
                # data loss (Y-scaled)
                Upred = model(t_b, x_b)
                L_data = ((Upred - U_b) * Y_inv).pow(2).mean()
                # PDE loss
                if step % cfg.pde_every_k_batches == 0:
                    t_c_np = np.random.uniform(t_start, t_end, size=cfg.n_coll)
                    x_c_np = np.random.uniform(x_min,   x_max,   size=cfg.n_coll)
                    keep   = _smoothness_mask(fem, t_c_np, x_c_np, cfg.grad_quantile)
                    t_c_np = t_c_np[keep]; x_c_np = x_c_np[keep]
                    if t_c_np.size < 32:
                        L_pde = torch.zeros((), device=device)
                    else:
                        t_c = torch.from_numpy(t_c_np).float().to(device).view(-1,1)
                        x_c = torch.from_numpy(x_c_np).float().to(device).view(-1,1)
                        t_c.requires_grad_(True); x_c.requires_grad_(True)
                        R = pde_residual(model, t_c, x_c, fem.gamma) * Y_inv
                        R = torch.clamp(R, -cfg.res_clip, cfg.res_clip)
                        L_pde = R.pow(2).mean()
                else:
                    L_pde = torch.zeros((), device=device)
                L_total = w_data * L_data + w_pde * L_pde
                if not torch.isfinite(L_total):
                    L_total = w_data * L_data
                optim.zero_grad(set_to_none=True)
                L_total.backward()
                all_finite = True
                for p in model.parameters():
                    if p.grad is not None and not torch.isfinite(p.grad).all():
                        all_finite = False; break
                if all_finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                    optim.step()
                ep_total += float(L_total.detach())
                ep_data  += float(L_data.detach())
                ep_pde   += float(L_pde.detach())
                end_epoch = time.time()
            ep_total /= steps_per_epoch
            ep_data  /= steps_per_epoch
            ep_pde   /= steps_per_epoch
            sched.step(ep_total)
            history["epoch"].append(epoch)
            history["L_total"].append(ep_total)
            history["L_data"].append(ep_data)
            history["L_pde"].append(ep_pde)
            history["w_data"].append(w_data)
            history["w_pde"].append(w_pde)
            history["lr"].append(optim.param_groups[0]["lr"])
            if ep_total < best_loss:
                best_loss = ep_total
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if epoch % log_every == 0 or epoch == cfg.epochs - 1:
                print(f"epoch {epoch:5d} | L_tot {ep_total:.3e} "
                      f"| L_data {ep_data:.3e} | L_pde {ep_pde:.3e} "
                      f"| w_data {w_data:.2f} w_pde {w_pde:.2f} "
                      f"| lr {optim.param_groups[0]['lr']:.1e}"
                      f"| t_elapsed {(end_epoch - start_time)/60:.2f} min")
        if best_state is not None:
            model.load_state_dict(best_state)

        # ----- write cache -----
        try:
            torch.save({
                "pinn_fingerprint": target_fp,
                "fem_fingerprint":  fem.fingerprint,
                "train_config":     trainconfig_to_dict(cfg),
                "model_state":      {k: v.detach().cpu()
                                     for k, v in model.state_dict().items()},
                "history":          history,
            }, cache_path)
            print(f"[INFO] PINN cache saved: {cache_path} "
                  f"(fingerprint={target_fp})")
        except Exception as exc:
            print(f"[WARN] could not save PINN cache: {exc}")

    # ----- evaluation @ terminal time -----
    T = fem.t_final
    xpoints = np.linspace(0.0, 1.0, 1000)
    model.eval()
    with torch.no_grad():
        t_q = torch.full((xpoints.size, 1), T, dtype=torch.float32, device=device)
        x_q = torch.from_numpy(xpoints).float().view(-1, 1).to(device)
        U_pinn = model(t_q, x_q).cpu().numpy()
    rho_p, q_p, E_p = U_pinn[:,0], U_pinn[:,1], U_pinn[:,2]
    u_p   = q_p / np.maximum(rho_p, 1e-12)
    p_p   = (fem.gamma - 1.0) * (E_p - 0.5 * rho_p * u_p**2)

    rho_fem = np.interp(xpoints, fem.x, fem.U[-1, :, 0])
    q_fem   = np.interp(xpoints, fem.x, fem.U[-1, :, 1])
    E_fem   = np.interp(xpoints, fem.x, fem.U[-1, :, 2])
    u_fem   = q_fem / np.maximum(rho_fem, 1e-12)
    p_fem   = (fem.gamma - 1.0) * (E_fem - 0.5 * rho_fem * u_fem**2)
    rho_ex, u_ex, p_ex = exact_riemann(xpoints, T, gamma_v=fem.gamma)

    err = dict(
        rho_fem_L2 =float(np.sqrt(np.mean((rho_fem - rho_ex)**2))),
        rho_pinn_L2=float(np.sqrt(np.mean((rho_p   - rho_ex)**2))),
        u_fem_L2   =float(np.sqrt(np.mean((u_fem   - u_ex  )**2))),
        u_pinn_L2  =float(np.sqrt(np.mean((u_p     - u_ex  )**2))),
        p_fem_L2   =float(np.sqrt(np.mean((p_fem   - p_ex  )**2))),
        p_pinn_L2  =float(np.sqrt(np.mean((p_p     - p_ex  )**2))),
    )

    # ----- comparison plot -----
    fig, ax = plt.subplots(2, 2, figsize=(11.5, 8))
    e_ex  = p_ex  / ((fem.gamma - 1.0) * rho_ex)
    e_fem = p_fem / ((fem.gamma - 1.0) * np.maximum(rho_fem, 1e-12))
    e_p   = p_p   / ((fem.gamma - 1.0) * np.maximum(rho_p,   1e-12))
    panels = [(ax[0,0], rho_ex, rho_fem, rho_p, r'$\rho$', 'Density'),
              (ax[0,1], u_ex,   u_fem,   u_p,   'u',       'Velocity'),
              (ax[1,0], p_ex,   p_fem,   p_p,   'p',       'Pressure'),
              (ax[1,1], e_ex,   e_fem,   e_p,   'e',       'Specific internal energy')]
    for axi, ex, fe, pi, ylab, title in panels:
        axi.plot(xpoints, ex, 'k-',  lw=1.5, label='Exact')
        axi.plot(xpoints, fe, 'g--', lw=1.0, label=r'SUPG-YZ$\beta$')
        axi.plot(xpoints, pi, 'r-',  lw=1.0, label='PASSC')
        axi.set_xlabel('x'); axi.set_ylabel(ylab); axi.set_title(title)
        axi.legend(); axi.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_plot, dpi=600, bbox_inches='tight')
    plt.close(fig)

    # ----- training diagnostics (only if we actually have a history) -----
    if history.get("epoch"):
        fig2, ax2 = plt.subplots(1, 2, figsize=(11, 4))
        ep = history["epoch"]
        ax2[0].semilogy(ep, history["L_total"], 'b-',  label='Total', lw=1)
        ax2[0].semilogy(ep, history["L_data"],  'g--', label='Data',  lw=1)
        ax2[0].semilogy(ep, [max(v, 1e-12) for v in history["L_pde"]],
                        'r:', label='PDE', lw=1)
        ax2[0].set_xlabel('epoch'); ax2[0].set_ylabel('loss')
        ax2[0].set_title('Loss components'); ax2[0].grid(alpha=0.3); ax2[0].legend()
        ax2[1].plot(ep, history["w_data"], 'g-',  label=r'$w_{data}$', lw=1.5)
        ax2[1].plot(ep, history["w_pde"],  'r--', label=r'$w_{pde}$',  lw=1.5)
        ax2[1].set_xlabel('epoch'); ax2[1].set_ylabel('weight')
        ax2[1].set_title('Adaptive weight schedule')
        ax2[1].grid(alpha=0.3); ax2[1].legend()
        fig2.tight_layout()
        fig2.savefig(out_plot.replace('.png', '_training.png'),
                     dpi=600, bbox_inches='tight')
        plt.close(fig2)
    else:
        print("[INFO] No training history available "
              "(model came from cache without history); skipping diagnostics plot.")

    print("\n--- L2 errors at t = T against the exact Riemann solution ---")
    for k, lab in (("rho","density"), ("u","velocity"), ("p","pressure")):
        fe, pi = err[f'{k}_fem_L2'], err[f'{k}_pinn_L2']
        ratio = fe / max(pi, 1e-15)
        print(f"  {lab:9s} :  SUPG-YZβ = {fe:.4e}   "
              f"PASSC = {pi:.4e}   ratio = {ratio:.2f}x")
    print(f"\nFigures: {out_plot}, {out_plot.replace('.png','_training.png')}")
    return err


# ===========================================================================
#                   CLI auto-generation from TrainConfig
# ===========================================================================

_CLI_SCALAR_TYPES = (int, float, str)
_CLI_SCALAR_TYPE_MAP = {"int": int, "float": float, "str": str}


def _is_cli_scalar(f) -> bool:
    """True if the dataclass field is a CLI-exposable scalar (int/float/str)."""
    if isinstance(f.type, type):
        return f.type in _CLI_SCALAR_TYPES
    return f.type in _CLI_SCALAR_TYPE_MAP


def _resolve_field_type(f):
    """Return the actual type object for a dataclass field (handles PEP 563)."""
    if isinstance(f.type, type):
        return f.type
    return _CLI_SCALAR_TYPE_MAP[f.type]


def _add_trainconfig_args(parser: argparse.ArgumentParser) -> None:
    """
    Auto-generate one CLI flag per scalar TrainConfig field.

    Non-scalar fields (e.g. `schedule`, which is a list of Phase objects) are
    intentionally skipped -- edit them in code rather than via the CLI.
    """
    for f in fields(TrainConfig):
        if not _is_cli_scalar(f):
            continue
        flag    = "--" + f.name.replace("_", "-")
        ftype   = _resolve_field_type(f)
        help_md = f.metadata.get("help", "")
        parser.add_argument(
            flag,
            type=ftype,
            default=f.default,
            dest=f.name,
            help=f"{help_md} (default: {f.default})",
        )


def _trainconfig_from_args(args: argparse.Namespace) -> TrainConfig:
    """Build a TrainConfig from the parsed CLI namespace."""
    kwargs = {f.name: getattr(args, f.name)
              for f in fields(TrainConfig)
              if _is_cli_scalar(f) and hasattr(args, f.name)}
    return TrainConfig(**kwargs)


# ===========================================================================
#                                   CLI
# ===========================================================================

def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Sod shock tube : SUPG-YZβ FEM + PASSC-Transient PINN.")

    # ----- Pipeline flags (not part of TrainConfig) -----
    p.add_argument("--skip-fem",  action="store_true",
                   help="Never run FEM; fail if no usable snapshot exists.")
    p.add_argument("--skip-pinn", action="store_true",
                   help="Run FEM only; do not train or evaluate the PINN.")
    p.add_argument("--force-fem", action="store_true",
                   help="Recompute FEM even if cache is valid.")
    p.add_argument("--force-pinn", action="store_true",
                   help="Retrain PINN even if cache is valid.")
    p.add_argument("--snapshots", default=SNAPSHOT_FILE,
                   help="Path to the FEM snapshot bundle.")
    p.add_argument("--pinn-cache", default=PINN_CACHE,
                   help="Path to the PINN weight cache.")
    p.add_argument("--device",    default="auto", choices=("auto", "cpu", "cuda"),
                   help="Compute device for the PINN step.")
    p.add_argument("--out",       default=PASSC_PLOT,
                   help="Output plot path for the PINN comparison.")

    # ----- Hyperparameter flags (auto-generated from TrainConfig) -----
    _add_trainconfig_args(p)

    args = p.parse_args(argv)

    # ----- STEP 1 : FEM (with cache) -----
    if args.skip_fem:
        if not os.path.exists(args.snapshots):
            print(f"[ERROR] --skip-fem set but {args.snapshots} does not exist.",
                  file=sys.stderr)
            return 2
        print("\n========  STEP 1 : FEM  ========")
        print(f"[INFO] --skip-fem set; reusing {args.snapshots} as-is.")
    else:
        print("\n========  STEP 1 : FEM  ========")
        if args.force_fem:
            print("[INFO] --force-fem set: recomputing FEM.")
            run_fem(snapshot_path=args.snapshots)
        elif fem_cache_is_valid(args.snapshots):
            print(f"[INFO] FEM cache hit: {args.snapshots} "
                  f"matches current settings (fingerprint={fem_fingerprint()}). "
                  f"Skipping FEM.")
        else:
            if os.path.exists(args.snapshots):
                print(f"[INFO] FEM cache present at {args.snapshots} but "
                      f"fingerprint differs; recomputing.")
            else:
                print(f"[INFO] No FEM cache at {args.snapshots}; computing.")
            run_fem(snapshot_path=args.snapshots)

    # ----- STEP 2 : PINN (with cache) -----
    if not args.skip_pinn:
        print("\n========  STEP 2 : PINN  ========")
        cfg = _trainconfig_from_args(args)
        run_pinn(snapshot_path=args.snapshots, cfg=cfg,
                 device_str=args.device, out_plot=args.out,
                 cache_path=args.pinn_cache,
                 force_retrain=args.force_pinn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
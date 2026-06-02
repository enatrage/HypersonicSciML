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
Claude
@author: scengizci  (FEM)  +   (PASSC post-processor)
"""

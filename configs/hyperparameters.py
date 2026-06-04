from dataclasses import dataclass, field
from typing import List

# ===========================================================================
#                       PROBLEM CONSTANTS (FEM & PINN)
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

SNAPSHOT_FILE  = "runs/fem/refactor/test1/sod_fem_snapshots.npz"
FEM_PLOT       = "runs/fem/refactor/test1/sod_cyz2.png"
PINN_CACHE     = "runs/pinn/refactor/test/sod_passc_model.pt"
PASSC_PLOT     = "runs/pinn/refactor/test/sod_passc.png"

# Reference scales used by YZβ and by the PINN's Y-scaled losses.
U1_REF_V = float(RHO_L_V)
U2_REF_V = float(RHO_L_V * 1.0)                      # ~ρ_L * a_L
U3_REF_V = float(P_L_V / (GAMMA_V - 1.0) + 0.5 * RHO_L_V * 1.0)      # ~E_L


# ===========================================================================
#                       TRAINING CONFIGURATION
# ===========================================================================
@dataclass
class Phase:
    upto_epoch: int
    w_data: float
    w_pde: float

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
    schedule: List[Phase] = field(default_factory=lambda: [
        Phase(upto_epoch=1000, w_data=1.00, w_pde=0.1),
        Phase(upto_epoch=2000, w_data=0.60, w_pde=0.60),
        Phase(upto_epoch=3500, w_data=0.30, w_pde=5.00),
        Phase(upto_epoch=10**9, w_data=0.01, w_pde=15.00),   # physics-dominant
    ])
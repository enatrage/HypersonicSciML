import os
import math
import torch
import numpy as np
from typing import Optional

from configs.hyperparameters import TrainConfig, SNAPSHOT_FILE, PINN_CACHE
from src.training.data import FemSnapshots, smoothness_mask
from src.model.pinn import build_model_from_cfg
from src.model.physics import pde_residual
from src.utils.hashing import pinn_fingerprint, pinn_cache_is_valid, trainconfig_to_dict

def run_pinn(snapshot_path: str = SNAPSHOT_FILE,
             cfg: Optional[TrainConfig] = None,
             device_str: str = "auto",
             cache_path: str = PINN_CACHE,
             force_retrain: bool = False) -> None:
    """
    Train (or load from cache) the PASSC-Transient PINN on FEM snapshots.
    """
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"snapshot file not found: {snapshot_path}")

    if cfg is None:
        cfg = TrainConfig()

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
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
          
    target_fp = pinn_fingerprint(cfg, fem.fingerprint)
    print(f"[INFO] PINN fingerprint = {target_fp}")

    U_ref = torch.from_numpy(fem.U_ref).float().to(device)
    Y_inv = (1.0 / U_ref).view(1, 3)

    use_cache = (not force_retrain) and pinn_cache_is_valid(cache_path, target_fp)
    history: dict = {}
    
    if use_cache:
        print(f"[INFO] PINN cache hit: loading {cache_path}")
        ckpt = torch.load(cache_path, map_location=device, weights_only=False)
        model = build_model_from_cfg(cfg, device)
        model.load_state_dict(ckpt["model_state"])
        return

    print(f"[INFO] No valid PINN cache at {cache_path}; commencing training.")

    K_s, Nx = fem.K_s, fem.Nx
    tt = np.broadcast_to(fem.t[:, None], (K_s, Nx)).reshape(-1, 1)
    xx = np.broadcast_to(fem.x[None, :], (K_s, Nx)).reshape(-1, 1)
    UU = fem.U.reshape(-1, 3)
    
    tt = torch.from_numpy(tt).float().to(device)
    xx = torch.from_numpy(xx).float().to(device)
    UU = torch.from_numpy(UU).float().to(device)

    t_start = float(fem.t.min()); t_end = float(fem.t.max())
    x_min   = float(fem.x.min()); x_max = float(fem.x.max())

    model = build_model_from_cfg(cfg, device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                              weight_decay=cfg.weight_decay,
                              betas=(0.9, 0.999))
                              
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim, mode='min', factor=0.9, patience=150, min_lr=1.0e-6)

    history = {k: [] for k in ("epoch","L_total","L_data","L_pde","w_data","w_pde","lr")}
    n_data = tt.shape[0]
    steps_per_epoch = max(1, math.ceil(n_data / cfg.batch_size))
    best_state, best_loss = None, float("inf")

    def _weights(epoch):
        for ph in cfg.schedule:
            if epoch < ph.upto_epoch: return ph.w_data, ph.w_pde
        return cfg.schedule[-1].w_data, cfg.schedule[-1].w_pde

    log_every = max(1, cfg.epochs // 25)
    
    for epoch in range(cfg.epochs):
        w_data, w_pde = _weights(epoch)
        perm = torch.randperm(n_data, device=device)
        ep_total = ep_data = ep_pde = 0.0
        
        for step in range(steps_per_epoch):
            sel = perm[step*cfg.batch_size:(step+1)*cfg.batch_size]
            t_b, x_b, U_b = tt[sel], xx[sel], UU[sel]
            
            Upred = model(t_b, x_b)
            L_data = ((Upred - U_b) * Y_inv).pow(2).mean()
            
            if step % cfg.pde_every_k_batches == 0:
                t_c_np = np.random.uniform(t_start, t_end, size=cfg.n_coll)
                x_c_np = np.random.uniform(x_min,   x_max,   size=cfg.n_coll)
                keep   = smoothness_mask(fem, t_c_np, x_c_np, cfg.grad_quantile)
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
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % log_every == 0 or epoch == cfg.epochs - 1:
            print(f"epoch {epoch:5d} | L_tot {ep_total:.3e} "
                  f"| L_data {ep_data:.3e} | L_pde {ep_pde:.3e} "
                  f"| w_data {w_data:.2f} w_pde {w_pde:.2f} "
                  f"| lr {optim.param_groups[0]['lr']:.1e}")
                  
    if best_state is not None:
        model.load_state_dict(best_state)

    try:
        torch.save({
            "pinn_fingerprint": target_fp,
            "fem_fingerprint":  fem.fingerprint,
            "train_config":     trainconfig_to_dict(cfg),
            "model_state":      {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "history":          history,
        }, cache_path)
        print(f"[INFO] PINN cache saved: {cache_path} (fingerprint={target_fp})")
    except Exception as exc:
        print(f"[WARN] could not save PINN cache: {exc}")
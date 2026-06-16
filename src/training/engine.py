import os
import math
import torch
import numpy as np
from typing import Optional
import time

from configs.hyperparameters import TrainConfig, SNAPSHOT_FILE, PINN_CACHE, NX, build_component_setups
from src.training.data import get_data
from src.model.physics import pde_residual
from src.utils.ml_hashing import pinn_fingerprint, pinn_cache_is_valid, trainconfig_to_dict
from src.utils.plotting import plot_pinn_graphs
from src.ops.loaders import build_model, build_optimizer, build_scheduler, build_loss

def run_pinn(snapshot_path: str = SNAPSHOT_FILE,
             cfg: Optional[TrainConfig] = None,
             device_str: str = "cuda",
             cache_path: str = PINN_CACHE,
             force_retrain: bool = False) -> None:
    """
    Train (or load from cache) the PASSC-Transient PINN on FEM snapshots.
    """


    ## INITIALS
    # Check snapshot
    if not os.path.exists(snapshot_path):
        raise FileNotFoundError(f"snapshot file not found: {snapshot_path}")
    # Get cfg
    if cfg is None:
        cfg = TrainConfig()
    # Build the <system>_setup dicts that drive the dynamic component loaders
    setups = build_component_setups(cfg)
    # Get seed
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    # Get device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"[INFO] PINN device = {device}")


    ## DATA
    # Get the dataloaders
    iterator = get_data(
        snapshot_path=snapshot_path,
        batch_size=cfg.batch_size,
        grad_quantile=cfg.grad_quantile,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    # Extract the dataset from the iterator, and the related properties 
    dataset = iterator.dataset
    n_tb = dataset.n_tb; K_s = dataset.K_s; Nx = dataset.Nx; gamma = dataset.gamma; data_fp = dataset.fp; t_final = dataset.t_final
    Y_inv = dataset.Y_inv
    print(f"[INFO] {K_s} snapshots, {Nx} nodes, gamma = {gamma}")
          

    ## MODEL RELATED
    # Gets the model, loss, and the optimizer, either from cache or freshly initialized
    # Model
    model_fp = pinn_fingerprint(cfg, data_fp)
    print(f"[INFO] PINN fingerprint = {model_fp}")
    use_cache = (not force_retrain) and pinn_cache_is_valid(cache_path, model_fp)
    if use_cache:
        print(f"[INFO] PINN cache hit: loading {cache_path}")
        ckpt = torch.load(cache_path, map_location=device, weights_only=False)
        model = build_model(setups["model_setup"], device)
        model.load_state_dict(ckpt["model_state"])
        return
    print(f"[INFO] No valid PINN cache at {cache_path}; commencing training.")
    model = build_model(setups["model_setup"], device)
    # Optim and schedule
    optim = build_optimizer(model.parameters(), setups["optim_setup"])
    lr_schedule = build_scheduler(optim, setups["scheduler_setup"])
    # Loss
    t_axis = dataset.t.view(K_s, Nx)[:, 0] # Initting related to collocation points, inputted to the loss
    x_axis = dataset.x.view(K_s, Nx)[0, :]
    smooth_map = dataset.is_smooth.view(K_s, Nx)
    t_start = float(dataset.t.min())
    loss_fn = build_loss(
        setups["loss_setup"], device,
        pde_residual_fn=pde_residual, gamma=gamma, Y_inv=Y_inv,
        t_axis=t_axis, x_axis=x_axis, smooth_map=smooth_map,
        t_start=t_start, t_final=t_final,
    )


    ## TRAINING RUN ITSELF
    history: dict = {}
    history = {k: [] for k in ("epoch","L_total","L_data","L_pde","w_data","w_pde","lr")}
    best_state, best_loss = None, float("inf")

    log_every = max(1, cfg.log_every_n) 

    start_time = time.time()
    for epoch in range(cfg.epochs):
        
        # Get the loss weights for this epoch, and reset the epoch cumulative losses
        w_data, w_pde = _weights(epoch, cfg)
        ep_total = ep_data = ep_pde = 0.0

        for step in range(n_tb):
            optim.zero_grad(set_to_none=True) # Zero the grad first

            t_in, x_in, U_target = next(iterator) # Get the data and move to gpu
            t_in = t_in.to(device)
            x_in = x_in.to(device)
            U_target = U_target.to(device)

            U_pred = model(t_in, x_in) # Get the predicted U

            # Delegate sub-graph construction and loss calculation to the module, collocation done inside the loss
            L_total, L_data_val, L_pde_val = loss_fn(
                model=model, U_pred=U_pred, U_target=U_target, 
                w_data=w_data, w_pde=w_pde, step=step
            )

            if cfg.debug_mode:
                # Fail-fast anomaly detection for mathematical instabilities
                with torch.autograd.detect_anomaly():
                    L_total.backward()
            else:
                L_total.backward()
            
            # Usefull in all cases for debug
            for name, param in model.named_parameters():
                if not torch.isfinite(param.grad).all():
                    print(f"NaN/Inf detected in gradients for {name}")
                    raise ValueError("Gradient explosion detected")

            # Clip the grads and take an optim step
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
                
            # Aggregate standard floats to prevent autograd memory leaks
            ep_total += L_total.item()
            ep_data  += L_data_val
            ep_pde   += L_pde_val
        
        # Logging procedure
        end_epoch = time.time()   
        ep_total /= n_tb
        ep_data  /= n_tb
        ep_pde   /= n_tb
        lr_schedule.step(ep_total)
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
                  f"| lr {optim.param_groups[0]['lr']:.1e}"
                  f"| t_elapsed {(end_epoch - start_time)/60:.2f} min")       
    

    ## MISC, SAVING

    # Pick the best case of the model, try to save it
    if best_state is not None:
        model.load_state_dict(best_state)
    try:
        torch.save({
            "pinn_fingerprint": model_fp,
            "fem_fingerprint":  data_fp,
            "train_config":     trainconfig_to_dict(cfg),
            "model_state":      {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "history":          history,
        }, cache_path)
        print(f"[INFO] PINN cache saved: {cache_path} (fingerprint={model_fp})")
    except Exception as exc:
        print(f"[WARN] could not save PINN cache: {exc}")

    # If the run was complete, plot diagnostics
    if history.get("epoch"):
        xpoints = np.linspace(0.0, 1.0, NX+1)
        model.eval()
        with torch.no_grad():
            t_q = torch.full((xpoints.size, 1), t_final, dtype=torch.float32, device=device)
            x_q = torch.from_numpy(xpoints).float().view(-1, 1).to(device)
            U_pinn = model(t_q, x_q).cpu().numpy()
        rho_pinn, q_pinn, E_pinn = U_pinn[:,0], U_pinn[:,1], U_pinn[:,2]
        u_pinn   = q_pinn / np.maximum(rho_pinn, 1e-12)
        p_pinn   = (gamma - 1.0) * (E_pinn - 0.5 * rho_pinn * u_pinn**2)
        e_pinn   = p_pinn   / ((gamma - 1.0) * np.maximum(rho_pinn,   1e-12))
        pinn_package = (rho_pinn, u_pinn, p_pinn, e_pinn)
        plot_pinn_graphs(history, pinn_package)
    else:
        print("[INFO] No training history available "
              "(model came from cache without history); skipping diagnostics plot.")

def _weights(epoch, cfg):
    for ph in cfg.schedule:
        if epoch < ph.upto_epoch: return ph.w_data, ph.w_pde
    return cfg.schedule[-1].w_data, cfg.schedule[-1].w_pde
import gc
import wandb
import logging
import torch
import torch.nn as nn

from src.master.schemas import FemConfig
from src.model.schemas import TrainConfig, DataConfig, ExportConfig
from src.model.utils.misc import get_n_params, check_lrtype_ifstep, build_physics_package
from src.model.utils.plotting import plot_pred_comparison

def train_model(
        model: nn.Module, loss_fn: nn.Module, optimizer: torch.optim, lr_scheduler: torch.optim.lr_scheduler, 
        data_iterator: torch.utils.data.DataLoader, device: torch.device, 
        train_cfg: TrainConfig, data_cfg: DataConfig, export_cfg: ExportConfig, fem_cfg: FemConfig
    ):

    # Infer the important details
    n_tb = data_iterator.dataset.n_tb
    grad_clip = train_cfg.grad_clip
    debug_mode = train_cfg.debug_mode
    n_epochs = train_cfg.n_epochs

    # Infer the type of lr_step
    is_lr_stepbased = check_lrtype_ifstep(lr_scheduler)


    model.to(device) # Move the model to the device
    loss_fn.to(device) # To capture any buffers
    if hasattr(loss_fn, "model"): loss_fn.model.to(device) # Move the loss' model to the device, if using such a loss
    best_val_loss = float('inf') # We'll use this to save the best model
    loss_str_tracker = {'total': 'Total', 'data': 'Data', 'pde': 'PDE'}
     # Log the param amount for model
    wandb.log({
        "model/param#": get_n_params(model=model)
    })

    logging.info(f"Training has started, will train for {n_epochs} epochs")
    for epoch in range(n_epochs):

        model.train() # Set the model to training mode just in case some val gets added downstream
        total_losses = {k: 0.0 for k in loss_str_tracker.keys()}

        logging.info(f"Model: Epoch {epoch}, Training Step{f', LR: {lr_scheduler.get_last_lr()[0]:.8f}' if lr_scheduler else ''}")
        for _ in range(n_tb):

            optimizer.zero_grad(set_to_none=True) # Zero the grad first

            t_train, x_train, U_target = next(data_iterator) # get the batch, move to device
            t_train = t_train.to(device); x_train = x_train.to(device); U_target = U_target.to(device)

            U_pred = model(t_train, x_train) # Get pred
            loss_total, loss_data, loss_pde, w_data, w_pde = loss_fn( 
                U_pred=U_pred, U_target=U_target
            ) # Get loss

            if debug_mode:
                # Fail-fast anomaly detection for mathematical instabilities
                with torch.autograd.detect_anomaly():
                    loss_total.backward()
                # Checks for NaN/Inf grads
                for name, param in model.named_parameters():
                    if not torch.isfinite(param.grad).all():
                        logging.warning(f"Model: NaN/Inf detected in gradients for {name}, in epoch {epoch+1}, step {_+1}")
                    raise ValueError("Gradient explosion detected")
            else:
                # If not debugging, just put the fries in the bag bro
                loss_total.backward()
            
            # Clip grad and step
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Log the losses for this step
            step_losses = [('total', loss_total), ('data', loss_data), ('pde', loss_pde)]
            log_payload = {"epoch": epoch}; print_parts = [] # Init the log payload
            for key, tensor in step_losses: 
                if tensor is not None:
                    val = tensor # Gets the singular item
                    total_losses[key] += val # Accumulates it to total
                    log_payload[f'train/{key}_loss_batch'] = val # Adds to the WandB payload
                    print_parts.append(f'{loss_str_tracker[key]}_Loss: {val:.8f}') # Adds to the print string
            wandb.log(log_payload)
            wandb.log({"train/w_data": w_data, "train/w_pde": w_pde}) # Log the weights too

            # If LR Scheduler is step based, take a step
            if is_lr_stepbased:
                lr_scheduler.step(loss_total) # Step the scheduler based on the total loss
        
        # Save the best model based on the total loss
        if total_losses['total'] < best_val_loss:
            best_val_loss = total_losses['total']
            torch.save(model.state_dict(), export_cfg.model_save_path)
            logging.info(f"Model: Best model saved at epoch {epoch+1} with total loss {best_val_loss:.8f}")
            
        # Log average train at the end of each epoch results
        log_payload, print_parts = {}, []
        for key, display_name in loss_str_tracker.items():
            total = total_losses.get(key, 0)
            if total != 0:
                avg = total / n_tb
                if key == 'total':
                    wb_key = "general/total_average_loss"
                else:
                    wb_key = f"general_detailed/train_average_{key}_loss"
                log_payload[wb_key] = avg
                print_parts.append(f"Average {display_name} Loss: {avg:.8f}")
        wandb.log(log_payload)
        if print_parts:
            print_parts[0] = f"Model: Epoch {epoch} Train Losses: " + print_parts[0]
            logging.info(' | '.join(print_parts))
        logging.info(f"Model: Epoch {epoch}, weights at w_data: {w_data:.4f}, w_pde: {w_pde:.4f}")
        
        # If the LR Scheduler is epoch based, take a step; take a loss step either way
        if not is_lr_stepbased:
            lr_scheduler.step(total_losses['total']) # Step the scheduler based on the total loss
        loss_fn.step() # Step the loss function for the next epoch

        # Release all the cache from training, just in case, to avoid memory leaks
        gc.collect()
        torch.cuda.empty_cache()

    # Get the final model evaluation and plot
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(0.0, 1.0, data_iterator.dataset.Nx, device=device).view(-1, 1)
        t_test = torch.full_like(x_test, data_iterator.dataset.t_final)
        # Evaluate and transfer to CPU NumPy
        U_pred_test = model(t_test, x_test).cpu()
        test_plot_package = build_physics_package(U_pred_test, data_iterator.dataset.gamma)
    plot_pred_comparison(
        test_plot_package=test_plot_package, x_points=x_test.cpu(), plot_comp_path=export_cfg.plot_comp_path, 
        snapshot_path=data_cfg.snapshot_path, fem_cfg=fem_cfg
    )
    
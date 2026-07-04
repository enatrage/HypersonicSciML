import torch
import torch.nn as nn
from typing import Callable

class CheatLoss(nn.Module):
    def __init__(
            self, 
            model: nn.Module, 
            smooth_map: torch.Tensor,
            Y_inv: torch.Tensor,
            t_axis: torch.Tensor,
            x_axis: torch.Tensor,
            t_start: float,
            t_final: float,
            gamma: float,
            pde_type: str,
            n_col: int,
            res_clip: float,
            weight_schedule: list[list[float, float, int]]
    ):
        super().__init__()

        # To prevent it from being registered as a submodule
        self._model_ref = [model]

        # Register bunch of stuff as a buffer so it automatically syncs with the module's device
        self.register_buffer("Y_inv", Y_inv.clone().detach())
        self.register_buffer("t_axis", t_axis.clone().detach())
        self.register_buffer("x_axis", x_axis.clone().detach())
        self.register_buffer("smooth_map", smooth_map.clone().detach())

        # Get the PDE resid
        self.pde_residual_fn = _physics_loader(pde_type)

        # Weight scheduling related
        self.weight_schedule = weight_schedule
        self.register_buffer("step_counter", torch.tensor(0, dtype=torch.long))

        # Rest of the assignments
        self.n_col = n_col
        self.t_start = t_start
        self.t_final = t_final
        self.gamma = gamma
        self.res_clip = res_clip

    def forward(
            self,
            U_pred: torch.Tensor,
            U_target: torch.Tensor
    ):
        
        # If the model is not in the device, change it
        device = U_pred.device

        # Get the PDE loss
        #L_pde = self.get_pde_loss(device)
        # Get the data loss
        L_data = self.get_data_loss(U_pred, U_target)

        # Get the current weights from the schedule and return the weighted sum
        w_data, w_pde = self._yield_scales()
        return w_data * L_data + w_pde * 0, L_data, 0, w_data, w_pde


    def get_data_loss(self, U_pred: torch.Tensor, U_target: torch.Tensor) -> torch.Tensor:
        """
        Computes the data-driven MSE loss between the predicted and target conservative variables.
        """
        return ((U_pred - U_target) * self.Y_inv).pow(2).mean()

    def get_pde_loss(self, device: torch.device) -> torch.Tensor:
        with torch.no_grad():
                # Generate uniform continuous tensors directly on the GPU
                t_raw = torch.empty(self.n_col, 1, device=device).uniform_(self.t_start, self.t_final)
                x_raw = torch.empty(self.n_col, 1, device=device).uniform_(0.0, 1.0)
                
                # O(1) Broadcasted nearest-neighbor lookup
                t_idx = torch.argmin(torch.abs(self.t_axis.unsqueeze(0) - t_raw), dim=1)
                x_idx = torch.argmin(torch.abs(self.x_axis.unsqueeze(0) - x_raw), dim=1)
                
                # Filter out continuous points that land near the discontinuous shock front
                keep = self.smooth_map[t_idx, x_idx]
                
                # Attach autograd tracking only to the surviving smooth points
                t_c = t_raw[keep].requires_grad_(True)
                x_c = x_raw[keep].requires_grad_(True)
        if t_c.shape[0] < 16:
            L_pde = torch.zeros((), device=device)
        else:
            R = self.pde_residual_fn(self.model, t_c, x_c, self.gamma) * self.Y_inv
            R = torch.clamp(R, -self.res_clip, self.res_clip)
            L_pde = R.pow(2).mean()
        return L_pde


    # This function needs to be called externally to increment the step counter for the loss scheduler
    def step(self):
        self.step_counter += 1

    def _yield_scales(self):
        current_step = self.step_counter.item()
        for w_data, w_pde, threshold_step in self.weight_schedule:
            if current_step < threshold_step: return w_data, w_pde
        # Fallback: if training continues past the final threshold, clamp to the final weights
        final_w_data, final_w_pde, _ = self.weight_schedule[-1]
        return final_w_data, final_w_pde

    @property
    def model(self):
        return self._model_ref[0]

def _physics_loader(pde_type: str) -> Callable:
    """
    Returns the appropriate physics residual function based on the choice string.
    """

    if pde_type == "euler":
        from src.model.architectures.physics import pde_residual_euler
        return pde_residual_euler
    else:
        raise ValueError(f"Unknown physics residual choice: {pde_type}")
    

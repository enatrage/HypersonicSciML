import torch
import torch.nn as nn

def euler_fluxes(U: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Computes the 1D Euler fluxes F(U).
    U shape: (N, 3) representing (rho, q, E) where q = rho * u.
    """
    rho = U[:, 0:1]
    q = U[:, 1:2]
    E = U[:, 2:3]
    
    # Floor density to prevent division by zero or NaN propagation in backprop
    rho_safe = rho.clamp(min=1.0e-8)
    u = q / rho_safe
    p = (gamma - 1.0) * (E - 0.5 * q * u)
    
    # F = [rho*u, rho*u^2 + p, (E + p)*u]
    return torch.cat([q, q * u + p, (E + p) * u], dim=-1)


def pde_residual(model: nn.Module, t: torch.Tensor, x: torch.Tensor, gamma: float) -> torch.Tensor:
    """
    Computes the strong-form residual of the 1D Euler equations.
    t, x must have requires_grad=True.
    """
    U = model(t, x)
    F = euler_fluxes(U, gamma)
    
    Ut_cols, Fx_cols = [], []
    for k in range(3):
        Uk = U[:, k:k+1]
        Fk = F[:, k:k+1]
        
        # dU/dt
        Ut_cols.append(torch.autograd.grad(
            Uk, t, grad_outputs=torch.ones_like(Uk),
            create_graph=True, retain_graph=True
        )[0])
        
        # dF/dx
        Fx_cols.append(torch.autograd.grad(
            Fk, x, grad_outputs=torch.ones_like(Fk),
            create_graph=True, retain_graph=True
        )[0])
        
    return torch.cat(Ut_cols, dim=-1) + torch.cat(Fx_cols, dim=-1)
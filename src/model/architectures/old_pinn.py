import torch
import torch.nn as nn

class FourierEmbedding(nn.Module):
    def __init__(self, input_dim: int, n_features: int, sigma: float):
        super().__init__()
        # B matrix is fixed during training
        B = torch.randn(input_dim, n_features) * sigma
        self.register_buffer("B", B)
        self.out_dim = input_dim + 2 * n_features

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        proj = z @ self.B
        return torch.cat([z, torch.sin(proj), torch.cos(proj)], dim=-1)

class ResidualBlock(nn.Module):
    def __init__(self, n_h: int):
        super().__init__()
        self.lin1 = nn.Linear(n_h, n_h)
        self.lin2 = nn.Linear(n_h, n_h)
        self.act  = nn.SiLU()
        self.norm = nn.LayerNorm(n_h)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        p = self.act(self.lin1(h))
        q = self.lin2(p)
        return self.act(self.norm(q + h))
    

class PINN_Euler(nn.Module):
    """
    Predicts u_NN(t, x) = (rho, q, E).
    rho and E are kept strictly positive via softplus reparametrisation.
    """
    def __init__(self, n_hidden: int = 48, n_blocks: int = 6, 
                 n_fourier: int = 24, sigma: float = 4.0,
                 rho_floor: float = 1.0e-3, E_floor: float = 1.0e-3):
        super().__init__()
        self.embed = FourierEmbedding(2, n_fourier, sigma)
        self.input_layer = nn.Linear(self.embed.out_dim, n_hidden)
        
        self.blocks = nn.ModuleList([ResidualBlock(n_hidden) for _ in range(n_blocks)])
        
        self.narrow = nn.Linear(n_hidden, n_hidden // 2)
        self.head   = nn.Linear(n_hidden // 2, 3)
        self.act    = nn.SiLU()
        
        self.rho_floor = rho_floor
        self.E_floor   = E_floor

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
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

def build_model_from_cfg(cfg, device: torch.device) -> PINN_Euler:
    """Instantiate a PINN_Euler with the given TrainConfig."""
    model = PINN_Euler(
        n_hidden=cfg.n_hidden, 
        n_blocks=cfg.n_blocks,
        n_fourier=cfg.n_fourier, 
        sigma=cfg.sigma,
    ).to(device)
    return model
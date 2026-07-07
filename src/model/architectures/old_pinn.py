import torch
import torch.nn as nn

class RandomTrigEmbedding(nn.Module):
    def __init__(self, i_dim: int=2, n_features: int=24, sigma: float=4.0):
        """
        A random trigonometric embedding layer (previously called "FourierEmbedding"), which artificially expands the expressivity in high-freq domains, via creating a
        [sin(zB), cos(zB)]

        Args:
            i_dim (int): Is the amount of spatio-temp dims, 2 at default for now
            n_features (int): The amount of randomized numbers for the amount of fourier features
            scale (float): The scaling for the randomized numbers for each feature+spatio-temporal dim

        Returns:
            torch.Tensor: The embedded data, with shape (B, i_dim + 2*n_features)

        Logic:
            z := (B, i_dim)
            pos_enc := randn(i_dim, n_features) * sigma
            proj = z @ pos_enc, (B, n_features)
            y = cat(z, sin(proj), cos(proj)), (B, [i_dim + n_feautures + n_feautres])
        """
        super().__init__()
        # pos_enc matrix is fixed during training
        pos_enc = torch.randn(i_dim, n_features) * sigma # with shape(i_dim, n_fourier)
        self.register_buffer("pos_enc", pos_enc)
        self.out_dim = i_dim + 2 * n_features # Define for later use

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z (torch.Tensor): With shape (B, i_dim)
        
        Returns:
            torch.Tensor: With shape (B, [i_dim + 2*n_feautres])
        """
        # z := (B, 2)
        proj = z @ self.pos_enc # Gets (B, n_features), positionally encodes z with the pre-determined encodings
        return torch.cat([z, torch.sin(proj), torch.cos(proj)], dim=-1) # Concat everything to be returned

class ResidualBlock(nn.Module):
    def __init__(self, h_dim: int):
        """
        Residual block used within the PINN_Euler

        Args:
            h_dim (int): The dimensions for linear layers
        
        Returns:
            torch.Tensor: The resid block passed tensor with shape (B, h_dim)

        Logic:
            x := Input tensor of shape (B, h_dim)
            forward_x = lin(act(lin(x))), with shape (B, h_dim)
            y = act(layernorm(x + forward_x)), with shape (B, h_dim)
        """
        super().__init__()

        self.layers_preresid = nn.Sequential(
            *[
                nn.Linear(h_dim, h_dim),
                nn.SiLU(),
                nn.Linear(h_dim, h_dim)
            ]
        )

        self.layers_postresid = nn.Sequential(
            *[
                nn.LayerNorm(h_dim),
                nn.SiLU()
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, h_dim)

        Returns:
            torch.Tensor: Output tensor with shape (B, h_dim)
        """
        # Logic presented in detail at __init__ docstring
        forward_x = self.layers_preresid(x)
        x = self.layers_postresid(forward_x + x)
        return x
    

class PINN_Euler(nn.Module):
    def __init__(
            self, 
            n_hidden: int = 48, 
            n_blocks: int = 6, 
            n_fourier: int = 24, 
            sigma: float = 4.0,
            rho_floor: float = 1.0e-7, 
            E_floor: float = 1.0e-7
    ):
        """
        PINN model to predict u_NN(t, x) = (rho, q, E). rho and E are kept strictly positive via softplus reparametrisation.

        Args:
            n_hidden (int): The amount of neurons in the intermittent residual blocks
            n_blocks (int): Amount of residual blocks
            n_fourier (int): Number of total "fourier" features in the random trigonometric embedding
            sigma (float): The stdev of the random trig embedding values
            rho_floor (float): Offset to ensure non-zero rho
            E_floor (floa): Offset to ensure non-zero E

        Returns:
            torch.Tensor: The predicted values returned as (B, [rho, q, E]) with shape (B, 3) with rho and q offset from zero

        Logic:
            x := (N, 1); t := (N, 1)
            x = act(linear(embed(cat(t, x)))), with shape (B, 2+2*n_fourier) embedded with features
            x = linear(x), with shape (B, n_hidden) projected for residually connected blocks
            x = residual**n_blocks(x), with shape (B, n_hidden) passed through the residual blocks n_blocks times
            x = act(linear(x)), with shape (B, n_hidden // 2) projected to a lower dim
            y = linear(x), with shape (B, 3) projected to final dims with the floors applied
        """
        super().__init__()

        # Create the rand trig embedding    
        embed_layer = RandomTrigEmbedding(2, n_fourier, sigma)
        layers = [
            embed_layer,
            nn.Linear(embed_layer.out_dim, n_hidden),
            nn.SiLU()
        ]
        # Unpack the residual blocks directly into the list
        layers.extend([ResidualBlock(n_hidden) for _ in range(n_blocks)])
        # Extend with final proj layer
        layers.extend([
            nn.Linear(n_hidden, n_hidden // 2),
            nn.SiLU(),
            nn.Linear(n_hidden // 2, 3)
        ])
        # Compile all into a single forward pass
        self.layers = nn.Sequential(*layers)

        self.rho_floor = rho_floor
        self.E_floor   = E_floor

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): Time tensor of shape (B, 1)
            x (torch.Tensor): Spatial tensor of shape (B, 1)

        Returns:
            torch.Tensor: The predicted rho, q, E values in order, all in a tensor of shape (B, 3)
        """
        assert t.shape == x.shape and x.ndim == 2 and x.shape[1] == 1, f"Invalid tensor input shape for t: {t.shape}, x: {x.shape};  must be (B,1) for both"

        # t= (B,1); x= (B,1)
        z = torch.cat([t, x], dim=-1) # z= (B, 2), this is the first input to the pipeline
        
        out = self.layers(z) # For detailed logic, check the docstring of __init__
        
        rho_out = out[:, 0:1] # Access the individual rows
        q = out[:, 1:2] # n:n+1 slicing to preserve tensor status
        E_out = out[:, 2:3] 
        rho = self.rho_floor + nn.functional.softplus(rho_out) # Apply the offset for numeric stability
        E = self.E_floor + nn.functional.softplus(E_out)
        
        return torch.cat([rho, q, E], dim=-1) # Return the concatted vals

import torch
import torch.nn as nn

class LAAF(nn.Module):
    """
    Learnable Adaptive Activation Function, recommended for theoretical benefit of enabling network to 
    have steeper activation functions that can be learned on spatial behavior. The base activation function
    is defined as SiLU for now. 

    Args:
        features (int): The number of features of the expected input tensor
        laaf_multip (float): Artificial scaling, the derivatives will carry a `n` multiplier, resulting in either faster or slower learning compared to other layers
    
    Returns:
        torch.Tensor: The tensor of the LAAF applied
    
    Logic:
        y = sigma(n * a `hadamard` x)
    """
    def __init__(self, features: int, laaf_multip: float = 10.0):
        super().__init__()
        # The 1/n allows for the acctivation to be equivalent to 1 at step 0
        self.a = nn.Parameter(torch.full((features, ), 1.0/laaf_multip)) 
        self.laaf_multip = laaf_multip
        self.act = nn.SiLU()
    
    def forward(self, x: torch.Tensor):
        return self.act(self.laaf_multip * self.a * x)

class LA_IrResidualBlock(nn.Module):
    def __init__(self, h_dim: int, laaf_multip: float, use_normed_layers: bool = False):
        """
        Residual block used within the LA_IrResPINN model, that uses learnable adaptive activation functions

        Args:
            h_dim (int): The dimensions for the layers
            laaf_multip (float): A scaling float for the LAAF activation functions
            use_normed_layers (bool): Boolean to either use norm on linear params or not, defaulting to false
        
        Returns:
            torch.Tensor: The resid block passed tensor with shape (B, h_dim)

        Logic:
            x := Input tensor of shape (B, h_dim)
            x_layer = lin(act(lin(x))), with shape (B, h_dim)
            y = x_layer + x, with shape (B, h_dim)
        """
        super().__init__()


        if use_normed_layers:
            self.layers_preresid = nn.Sequential(
                *[
                    nn.utils.weight_norm(nn.Linear(h_dim, h_dim)),
                    LAAF(h_dim, laaf_multip),
                    nn.utils.weight_norm(nn.Linear(h_dim, h_dim))
                ]
            )
        
        else:
            self.layers_preresid = nn.Sequential(
                *[
                    nn.Linear(h_dim, h_dim),
                    LAAF(h_dim, laaf_multip),
                    nn.Linear(h_dim, h_dim)
                ]
            )
            # Force block to have mathematical identity property
            nn.init.zeros_(self.layers_preresid[2].weight)
            nn.init.zeros_(self.layers_preresid[2].bias)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Input tensor with shape (B, h_dim)

        Returns:
            torch.Tensor: Output tensor with shape (B, h_dim)
        """
        # Logic presented in detail at __init__ docstring
        return self.layers_preresid(x)+x

class LA_IrResPINN(nn.Module):
    def __init__(
            self, 
            i_dim: int = 2, 
            o_dim: int = 3, 
            n_hidden: int = 64, 
            n_blocks: int = 10,
            laaf_multip: float = 10.0,
            use_normed_layers: bool = False,
            rho_floor: float = 1.0e-7, 
            E_floor: float = 1.0e-7
    ):
        """
        Locally Adaptive Identity Routed Residual PINN, that predicts model(x, t) = [rho, q, E]. Uses locally adaptive
        activation functions and identity routed residuals for non-destructive residual connections.

        Args:
            i_dim (int): The size of input dim, fixed at 2 [x, t] for the repo
            o_dim (int): The size of ouput dim, fixed at 3 [rho, q, E] for the repo
            n_hidden (int): Amount of neurons in each LA_IrResidualBlock
            n_blocks (int): The amount of total sequential LA_IrResidualBlocks
            laaf_multip (float): A scaling float for the LAAF activation functions
            use_normed_layers (bool): Boolean to turn on or off norming on linear layers, suggested as off for AdamW
            rho_floor (float): Offset to ensure non-zero rho
            E_floor (floa): Offset to ensure non-zero E
        
        Returns:
            torch.Tensor: The predicted values returned as (B, [rho, q, E]) with shape (B, 3), without any offset for rho and q

        Logic:
            x := (N, 1); t := (N, 1)
            x = linear(x), with shape (B, n_hidden) projected for residually connected blocks
            x = residual**n_blocks(x), with shape (B, n_hidden) passed through the residual blocks n_blocks times
            x = act(linear(x)), with shape (B, n_hidden // 2) projected to a lower dim
            y = linear(x), with shape (B, 3) projected to final dims, with high beta softplus applied to rho and E for non-negativity
        """
        super().__init__()

        self.rho_floor = rho_floor
        self.E_floor = E_floor

        # Initial projection and subsequent LAAF
        layers = [
            nn.Linear(i_dim, n_hidden),
            LAAF(n_hidden, laaf_multip)
        ]

        # Repeated hidden blocks
        layers.extend([LA_IrResidualBlock(n_hidden, laaf_multip, use_normed_layers) for _ in range(n_blocks)])

        # Final projection followed by LAAF and another projection
        layers.extend([
            nn.Linear(n_hidden, n_hidden // 2),
            LAAF(n_hidden // 2, laaf_multip),
            nn.Linear(n_hidden // 2, o_dim)
        ])

        self.layers = nn.Sequential(*layers)

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
        
        out = self.layers(z) 
        
        rho_out = out[:, 0:1] 
        q = out[:, 1:2] # q remains strictly linear
        E_out = out[:, 2:3] 
        
        # High-Beta Softplus: Acts as an identity function for positive values 
        # to preserve PDE gradients, while asymptotically bounding negatives.
        rho = self.rho_floor + nn.functional.softplus(rho_out, beta=10.0)
        E = self.E_floor + nn.functional.softplus(E_out, beta=10.0)
        
        return torch.cat([rho, q, E], dim=-1)


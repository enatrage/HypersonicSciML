from pydantic import BaseModel, Field
from typing import Union, Literal, Annotated

class OldPINNConfig(BaseModel):
    type: Literal["old_pinn"]="old_pinn"
    n_hidden: int = Field(48, description="Hidden layer width")
    n_blocks: int = Field(6, description="Number of residual blocks")
    n_fourier: int = Field(24, description="Fourier feature count")
    sigma: float = Field(4.0, description="Fourier feature scale")

class LAIrResPINNConfig(BaseModel):
    type: Literal["LAIrResPINN"]="LAIrResPINN"
    i_dim: int = Field(2, description="The size of input dim, fixed at 2 [x, t] for the repo")
    o_dim: int = Field(3, description="The size of ouput dim, fixed at 3 [rho, q, E] for the repo")
    n_hidden: int = Field(64, description="Amount of neurons in each LA_IrResidualBlock")
    n_blocks: int = Field(10, description="The amount of total sequential LA_IrResidualBlocks")
    laaf_multip: float = Field(10.0, description="A scaling float for the LAAF activation functions")
    use_normed_layers: bool = Field(False, description="Boolean to turn on or off norming on linear layers, suggested as off for AdamW")
    rho_floor: float = Field(1.0e-7, description="Offset to ensure non-zero rho")
    E_floor: float = Field(1.0e-7, description="Offset to ensure non-zero E")

ArchitectureConfig = Annotated[
    Union[OldPINNConfig, LAIrResPINNConfig], 
    Field(discriminator="type")
]
from pydantic import BaseModel, Field, computed_field
from typing import Union, Literal, List, Annotated, Optional

class OldPINNConfig(BaseModel):
    type: Literal["old_pinn"]="old_pinn"
    n_hidden: int = Field(48, description="Hidden layer width")
    n_blocks: int = Field(6, description="Number of residual blocks")
    n_fourier: int = Field(24, description="Fourier feature count")
    sigma: float = Field(4.0, description="Fourier feature scale")

ArchitectureConfig = Annotated[
    Union[OldPINNConfig], 
    Field(discriminator="type")
]
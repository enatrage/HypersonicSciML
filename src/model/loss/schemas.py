from pydantic import BaseModel, Field
from typing import Union, Literal, Annotated

class OldLossConfig(BaseModel):
    type: Literal["old_loss"]="old_loss"
    res_clip: float = Field(..., description="PDE residual clip value")
    n_col: int = Field(..., description="Collocation points per PDE step")

LossConfig = Annotated[
    Union[OldLossConfig],
    Field(discriminator="type")
]
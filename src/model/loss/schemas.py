from pydantic import BaseModel, Field
from typing import Union, Literal, Annotated

class CheatLossConfig(BaseModel):
    type: Literal["cheat_loss"]="cheat_loss"
    pde_type: Literal["euler"]="euler"
    n_col: int = Field(..., description="Collocation points per PDE step")
    res_clip: float = Field(..., description="PDE residual clip value")
    weight_schedule: list[list[float, float, int]] = Field(..., description="Weight schedule for the loss function. Each entry is a list of [w_data, w_pde, epoch]")

LossConfig = Annotated[
    Union[CheatLossConfig],
    Field(discriminator="type")
]
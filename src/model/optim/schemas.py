from pydantic import BaseModel, Field
from typing import Union, Literal, Tuple, Annotated, Optional

class AdamWConfig(BaseModel):
    type: Literal["adam_w"] = "adam_w"
    lr: float = Field(1.0e-3, description="Learning rate")
    betas: Tuple[float, float] = Field((0.9, 0.999), description="Coefficients used for computing running averages of gradient and its square")
    eps: float = Field(1.0e-8, description="Term added to the denominator to improve numerical stability")
    weight_decay: float = Field(1.0e-2, description="Weight decay coefficient. In AdamW, this is decoupled from the gradient update")
    amsgrad: bool = Field(False, description="Whether to use the AMSGrad variant of this algorithm")

class AdamConfig(BaseModel):
    type: Literal["adam"] = "adam"
    lr: float = Field(1.0e-3, description="Learning rate.")
    betas: Tuple[float, float] = Field((0.9, 0.999), description="Coefficients for running averages.")
    eps: float = Field(1.0e-8, description="Numerical stability term.")
    weight_decay: float = Field(0.0, description="L2 penalty. In standard Adam, this is coupled with the gradient update.")
    amsgrad: bool = Field(False, description="Use the AMSGrad variant.")

class SGDConfig(BaseModel):
    type: Literal["sgd"] = "sgd"
    lr: float = Field(1.0e-3, description="Learning rate")
    momentum: float = Field(0.0, description="Momentum factor")
    weight_decay: float = Field(0.0, description="Weight decay (L2 penalty)")
    dampening: float = Field(0.0, description="Dampening for momentum")
    nesterov: bool = Field(False, description="Enables Nesterov momentum. Requires momentum > 0 and dampening == 0")

class LBFGSConfig(BaseModel):
    type: Literal["lbfgs"] = "lbfgs"
    lr: float = Field(1.0, description="Learning rate")
    max_iter: int = Field(20, description="Max number of iterations per optimization step")
    max_eval: Optional[int] = Field(None, description="Max number of function evaluations per optimization step. Default is max_iter * 1.25")
    tolerance_grad: float = Field(1.0e-7, description="Termination tolerance on first order optimality")
    tolerance_change: float = Field(1.0e-9, description="Termination tolerance on function value/parameter changes")
    history_size: int = Field(100, description="Update history size")
    line_search_fn: Optional[Literal["strong_wolfe"]] = Field(None, description="Line search function to use. 'strong_wolfe' is highly recommended for stable PDE residual convergence")

OptimizerConfig = Annotated[
    Union[
        AdamWConfig, 
        AdamConfig, 
        SGDConfig, 
        LBFGSConfig
    ],
    Field(discriminator="type")
]
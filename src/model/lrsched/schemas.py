from pydantic import BaseModel, Field
from typing import Union, Literal, Annotated, Optional
from pydantic import model_validator

class ReduceLROnPlateauConfig(BaseModel):
    type: Literal["reduce_on_plateau"] = "reduce_on_plateau"
    mode: Literal["min", "max"] = Field("min", description="Quantity monitored: 'min' for loss residuals, 'max' for accuracy/metrics")
    factor: float = Field(0.1, description="Multiplicative factor of learning rate decay. new_lr = lr * factor")
    patience: int = Field(10, description="Number of epochs with no improvement after which learning rate will be reduced")
    threshold: float = Field(1.0e-4, description="Threshold for measuring the new optimum, to only focus on significant changes and ignore noise")
    min_lr: float = Field(0.0, description="A lower bound on the learning rate of all parameter groups")

class CosAnnConfig(BaseModel):
    type: Literal["cos_annealing"] = "cos_annealing"
    T_max: int = Field(..., description="Maximum number of iterations. Typically the total number of epochs or steps")
    eta_min: float = Field(0.0, description="Minimum learning rate reached at the bottom of the cosine curve")

class CosAnnWarmRestartsConfig(BaseModel):
    type: Literal["cos_ann_warm_restarts"] = "cos_ann_warm_restarts"
    T_0: int = Field(..., description="Number of iterations for the first restart")
    T_mult: int = Field(1, description="A factor increases T_{i} after a restart. T_{i} = T_{i-1} * T_mult")
    eta_min: float = Field(0.0, description="Minimum learning rate reached at the end of a cycle")

class OneCycleLRConfig(BaseModel):
    type: Literal["one_cycle"] = "one_cycle"
    max_lr: float = Field(..., description="Upper learning rate boundary in the cycle for each parameter group")
    total_steps: Optional[int] = Field(None, description="The total number of steps in the cycle. Note that if a value is not provided here, then it must be inferred by providing a value for epochs and steps_per_epoch")
    epochs: Optional[int] = Field(None, description="The number of epochs to train for. Used in conjunction with steps_per_epoch")
    steps_per_epoch: Optional[int] = Field(None, description="The number of steps per epoch to train for. Used in conjunction with epochs")
    pct_start: float = Field(0.3, description="The percentage of the cycle (in number of steps) spent increasing the learning rate")

    @model_validator(mode='after')
    def check_step_definitions(self) -> 'OneCycleLRConfig':
        has_total = self.total_steps is not None
        has_components = self.epochs is not None and self.steps_per_epoch is not None
        if not (has_total or has_components):
            raise ValueError("You must define either 'total_steps' OR ('epochs' AND 'steps_per_epoch')")
        return self

class StepConfig(BaseModel):
    type: Literal["step"] = "step"
    step_size: int = Field(..., description="Period of learning rate decay in epochs (or steps if called dynamically)")
    gamma: float = Field(0.1, description="Multiplicative factor of learning rate decay")

LRSchedConfig = Annotated[
    Union[
        CosAnnConfig, 
        StepConfig, 
        ReduceLROnPlateauConfig, 
        CosAnnWarmRestartsConfig, 
        OneCycleLRConfig
    ],
    Field(discriminator="type")
]

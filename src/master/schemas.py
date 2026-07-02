from pydantic import BaseModel, Field, computed_field

from src.model.lrsched.schemas import LRSchedConfig
from src.model.loss.schemas import LossConfig
from src.model.architectures.schemas import ArchitectureConfig
from src.model.optim.schemas import OptimizerConfig
from src.model.schemas import TrainConfig, DataConfig, ExportConfig

from src.fem.schemas import ThermodynamicsConfig, DomainConfig, InitialConditionsConfig, ScalingConfig, FemIOConfig

class ModelConfig(BaseModel):
    architecture: ArchitectureConfig
    loss: LossConfig 
    optimizer: OptimizerConfig
    lrsched: LRSchedConfig
    training: TrainConfig
    data: DataConfig
    export: ExportConfig

class FemConfig(BaseModel):
    thermodynamics: ThermodynamicsConfig
    domain: DomainConfig
    initial_conditions: InitialConditionsConfig
    scaling: ScalingConfig
    io: FemIOConfig

    @computed_field
    @property
    def u1_ref(self) -> float:
        return self.initial_conditions.left.rho
    
    @computed_field
    @property
    def u2_ref(self) -> float:
        # Represents ~ρ_L * a_L
        return self.initial_conditions.left.rho * 1.0
    
    @computed_field
    @property
    def u3_ref(self) -> float:
        # Represents the Total Energy E_L
        p_l = self.initial_conditions.left.p
        rho_l = self.initial_conditions.left.rho
        gamma = self.thermodynamics.gamma
        return (p_l / (gamma - 1.0)) + (0.5 * rho_l * 1.0)


class RunModeConfig(BaseModel):
    fem: bool = Field(..., description="Flag to execute FEM data generation stage")
    model: bool = Field(..., description="Flag to execute model training stage")

class MasterConfig(BaseModel):
    run_mode: RunModeConfig
    fem_config: FemConfig
    model_config: ModelConfig

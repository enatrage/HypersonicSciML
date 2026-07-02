from pydantic import BaseModel, Field

class TrainConfig(BaseModel):
    seed: int = Field(..., description="RNG seed")
    device: str = Field("cuda", description="Device for the training sequence, options are 'auto', 'cuda', 'cpu'")
    grad_clip: float = Field(..., description="Gradient-norm clip")
    debug_mode: bool = Field(..., description="Debug mode, yes/no, does anomaly detection for instabilities")
    n_epochs: int = Field(..., description="Number of epochs for the training process")

class DataConfig(BaseModel):
    snapshot_path: str = Field(..., description="The location for the snapshot to be loaded")
    num_workers: int = Field(description="Num of workers for the train dataloader, 0 as default since data is simple and small")
    pin_memory: bool = Field(description="Whether to pin the worker(s) in memory")
    batch_size: int = Field(description="Train batch size")
    grad_quantile: float = Field(description="Smoothness quantile threshold")

class ExportConfig(BaseModel):
    wandb_entity: str = Field(description="Entity for WandB log")
    wandb_project: str = Field(description="Project for WandB log")
    wandb_name: str = Field(description="Name for WandB log")
    local_log_path: str = Field(description="Path for the local log")
    model_save_path: str = Field(description="Path for the model save and checkpoints")
    plot_comp_path: str = Field(description="Path related to model's prediction comparison to data")
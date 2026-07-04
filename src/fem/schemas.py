from pydantic import BaseModel, Field

class StateConfig(BaseModel):
    rho: float = Field(..., gt=0.0, description="Macroscopic fluid density for the respective left or right state")
    u: float = Field(..., description="Macroscopic bulk velocity for the respective left or right state")
    p: float = Field(..., gt=0.0, description="Static thermodynamic pressure for the respective left or right state")

class InitialConditionsConfig(BaseModel):
    left: StateConfig
    right: StateConfig

class ThermodynamicsConfig(BaseModel):
    gamma: float = Field(..., gt=1.0, description="The ratio of specific heats (C_p/C_v). It acts as the adiabatic index closing the Euler equations via the ideal gas equation of state p=(gamma - 1)*rho*e")

class DomainConfig(BaseModel):
    x_diaph: float = Field(..., gt=0.0, lt=1.0, description="Spatial coordinate of the diaphragm")
    nx: int = Field(..., gt=0, description="Number of finite element cells (elements) in the mesh, #nodes=nx+1")
    t_final: float = Field(..., gt=0.0, description="Terminal time for the implicit solver")
    dt: float = Field(..., gt=0.0, description="Time step size for the backward Euler weak form")

class ScalingConfig(BaseModel):
    beta: float = Field(..., gt=0.0, description="Exponent beta for the YZ-beta shock capturing viscosity")
    k_s: int = Field(..., gt=0, description="Maximum rolling buffer size for retained snapshots saved to the archive")

class FemIOConfig(BaseModel):
    snapshot_path: str = Field(..., description="The location for the snapshot to be saved")
    plot_path: str = Field(..., description="The location for the FEM plot to be saved")
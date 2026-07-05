"""
The model (on top of the PINN_Euler) must fix the following:

1- No LayerNorm:
    Layernorm completely destroys the autograd and makes it incredibly oscillatory, additionally, it probably provides
    very diminishing returns for the problems it causes.
2- 
    
"""
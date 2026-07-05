"""
A simple forward pass test on the refactored old code, to ensure no tensor shape mismatches occur
"""

def test_function():

    from src.model.pinn import PINN_Euler
    import torch

    model = PINN_Euler()
    batch_size = 64

    dummy_x = torch.rand(size=(batch_size, 1))
    dummy_t = torch.rand(size=(batch_size, 1))
    out = model(dummy_t, dummy_x)

    assert out.shape == (batch_size, 3), f"Wrong output shape: {out.shape}, was expecting ({batch_size}, 3)"

if __name__ == "__main__":
    
    test_function()
"""
A simple forward pass test on the built architectures of models
"""

def load_model(choice: str):

    if choice == "old_pinn":
        from src.model.architectures.old_pinn import PINN_Euler
        return PINN_Euler()
    elif choice == "LAIrResPINN":
        from src.model.architectures.la_irrespinn import LA_IrResPINN
        return LA_IrResPINN()
    elif choice == "dual_path":
        from src.model.architectures.dual_path import DualPath
        return DualPath()
    else:
        raise ValueError(f"Unknown choice: {choice}")

def test_function(model):

    import torch
    batch_size = 64
    dummy_x = torch.rand(size=(batch_size, 1))
    dummy_t = torch.rand(size=(batch_size, 1))
    out = model(dummy_t, dummy_x)
    assert out.shape == (batch_size, 3), f"Wrong output shape: {out.shape}, was expecting ({batch_size}, 3)"

    return True

if __name__ == "__main__":

    choice = "LAIrResPINN"
    model = load_model(choice)

    if test_function(model):
        print("Test: Passed")
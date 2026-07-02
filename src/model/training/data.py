import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
from typing import Iterator, Union

class FemSnapshots(Dataset):
    """
    Pytorch dataset for loading FEM snapshots from a .npz file

    Args:
        data (np.ndarray): The FEM snapshots data loaded from a .npz file
        grad_quantile (float): The quantile threshold for determining smooth regions in the FEM solution, used for the PDE Smoothness Mask. Higher values retain more points as "smooth", lower values are more strict

    Returns:
        A PyTorch Dataset object that can be used with a DataLoader for training a neural network
    """
    def __init__(self, data: np.ndarray, U1_ref: float, U2_ref: float, U3_ref: float, grad_quantile: float):

        # Access the static physical and numerical params
        self.gamma = float(data["gamma"])
        self.t_start = float(data["t_start"])
        self.t_final = float(data["t_final"])
        self.dt = float(data["dt"])
        self.U_ref = torch.from_numpy(data["U_ref"].astype(np.float64)).float()

        # Extract raw spatio-temporal arrays
        x_raw = data["x"].astype(np.float64) # spatial array
        t_raw = data["t_snap"].astype(np.float64) # temporal array
        self.t_axis = torch.from_numpy(t_raw).float() # Store these for downstream use
        self.x_axis = torch.from_numpy(x_raw).float()
        U_raw = np.stack([data["rho_snap"], data["q_snap"], data["E_snap"]], axis=-1).astype(np.float64) # the raw value array
        self.K_s = t_raw.size # number of snapshots
        self.Nx = x_raw.size # number of cells

        # Pre-compute the PDE Smoothness Mask, data leakage here... if we were to be doing train/test/val split
        # For a "post-processing framework", it is fine
        dx_fem = x_raw[1] - x_raw[0]
        drho_dx = np.gradient(U_raw[..., 0], dx_fem, axis=1)
        abs_grad = np.abs(drho_dx)
        thresh = np.quantile(abs_grad[-1], grad_quantile)
        is_smooth_raw = ~(abs_grad > thresh)

        # Unroll into flat coord maps
        tt = np.broadcast_to(t_raw[:, None], (self.K_s, self.Nx)).reshape(-1, 1)
        xx = np.broadcast_to(x_raw[None, :], (self.K_s, self.Nx)).reshape(-1, 1)
        UU = U_raw.reshape(-1, 3)
        smooth = is_smooth_raw.reshape(-1)

        # The final accessible tensors
        self.t = torch.from_numpy(tt).float()
        self.x = torch.from_numpy(xx).float()
        self.U = torch.from_numpy(UU).float()
        self.is_smooth = torch.from_numpy(smooth).bool() # This is accessed by some downstream logic, even though not returned in __getitem__

        # Build the U_ref and the Y_inv factors
        # Reference scales used by YZβ and by the PINN's Y-scaled losses.
        self.U_ref = torch.tensor([U1_ref, U2_ref, U3_ref]).float()
        self.Y_inv = (1.0 / self.U_ref).view(1, 3)

    def __len__(self):
        # Total number of collocation points (K_s * Nx)
        return self.t.shape[0]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Returns a purely numerical tuple: (t_i, x_i, U_i)
        # Can also return is_smooth[idx] if required, but this is not used in the pipeline for now
        return self.t[idx], self.x[idx], self.U[idx]


def get_dataloaders(
        snapshot_path: Union[str, Path], 
        U1_ref: float, U2_ref: float, U3_ref: float,
        batch_size: int, 
        grad_quantile: float, 
        num_workers: int, 
        pin_memory: bool
    ) -> DataLoader:
    """
    The primary function to get the dataloaders for the training sequence

    Args:
        snapshot_path (str, Path): The path or string for the snapshot with which we are training the model
        U1-3_ref (float): The reference scales for the conserved variables, used for Y-scaling in the loss function
        batch_size (int): The batch size for the train dataloader, extra batches will be dropped!
        grad_quantile (float): The quantile threshold for determining smooth regions in the FEM solution
        num_workers (int): Num of workers to hold the dataset in after loading
        pin_memory (bool): Whether to pin the workers in the memory

    Returns:
        DataLoader: A created dataloader
    """
    # Unpack data
    data = _load_saved_data(snapshot_path)
    dataset = FemSnapshots(data, grad_quantile, U1_ref, U2_ref, U3_ref)

    # Build the dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False # Fixed to false
    ) 
    # append the number of training batch as a property to the dataset
    dataloader.dataset.n_tb = len(dataloader)
    
    return dataloader


def _load_saved_data(snapshot_path: Union[str, Path]):
    snapshot_path_str = str(snapshot_path)
    if snapshot_path_str.endswith(".npz"):
        data = np.load(snapshot_path_str)
        return data
    else:
        raise ValueError(f"Non-implemented data loading mechanism for the path: {snapshot_path_str}")

class InfiniteCycle:
    """
    A custom iterator wrapper that allows infinite cycling over a DataLoader
    while exposing the underlying dataset as a public attribute.
    """
    def __init__(self, dataloader: DataLoader):
        self._dataloader = dataloader
        self.dataset = dataloader.dataset
        self._iterator = self._generate()

    def _generate(self):
        while True:
            for batch in self._dataloader:
                yield batch

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iterator)


def _get_inf_iterator(dataloader: DataLoader) -> Iterator:
    """
    Gets the infinite iterator, it can be infinitely iterated through. 

    Args:
        dataloader (DataLoader): An already prepared dataloader
    
    Returns:
        iterator (Iterator): An infinitely callable iterator, with the dataset bound to it as an attribute
    """
    return InfiniteCycle(dataloader)

def get_data(snapshot_path: Union[str, Path], U1_ref: float, U2_ref: float, U3_ref: float, batch_size: int, grad_quantile: float, num_workers: int = 8, pin_memory: bool = True) -> Iterator:
    """
    The primary function to get the iterator for the training sequence

    Args:
        snapshot_path (str, Path): The path or string for the snapshot with which we are training the model
        U1-3_ref (float): The reference scale for the conserved variables, used for Y-scaling
        batch_size (int): The batch size for the train dataloader, extra batches will be dropped!
        grad_quantile (float): The quantile threshold for determining smooth regions in the FEM solution
        num_workers (int): Num of workers to hold the dataset in after loading
        pin_memory (bool): Whether to pin the workers in the memory
    
    Returns:
        Iterator: A list of the created dataloader iterators in train/val/test order, val and test being infinite iterators
    """

    loader= get_dataloaders(snapshot_path, U1_ref, U2_ref, U3_ref, batch_size, grad_quantile, num_workers, pin_memory)
    iterator = _get_inf_iterator(loader) # Finite iterator for training

    return iterator
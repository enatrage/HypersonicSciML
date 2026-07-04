# HypersonicSciML
Repository for the code used during my external collaboration to Prof. Süleyman Cengizci's TÜBİTAK 225M468 Project. In this project, we develop a SciML model to conduct post-processing and model training on hypersonic flows.

## Installation Guide
This project utilizes a hybrid environment provisioning strategy. The C++ physics backends (FEniCS, mshr) must be installed via Conda to avoid source-compilation failures, while the machine learning stack (PyTorch, WandB) is handled via Pip to ensure precise CUDA version matching.

**Prerequisites:** Linux or WSL2, Conda (Miniconda/Anaconda), and NVIDIA drivers correctly configured on the host machine.

**Step 1: Provision the Physics Backend**

Create and activate a Conda environment to fetch pre-compiled binaries for FEniCS, mshr, and their dependencies (PETSc, SLEPc, MPI).
```bash
conda create -n <name_of_venv> -c conda-forge fenics mshr python=3.10
conda activate <name_of_venv>
```

**Step 2: Install the Machine Learning Stack**

Ensure the repository's `requirements.txt` is present in the root directory (it must include the `--extra-index-url https://download.pytorch.org/whl/cu118` directive for PyTorch). Install the remaining stack via pip:
```bash
pip install -r requirements.txt --no-cache-dir
```

**Step 3: Verify the Environment**

Run the verification script to confirm that PyTorch has successfully captured the CUDA passthrough and that the FEniCS C++ bindings have not been overwritten.
```bash
python scripts/verify_env.py
```

## User Guide
- Activate your venv for the project and CD into the root
- Login to wandb via `wandb login <your_api_key>`
- Run `python -m scripts.master_script` with the following add-ons:
    - `--base-cfg` or `-bc` followed by the base config path (this is required)
    - `--addon-cfg` or `-ac` followed by the addon config path (optional, will override entries in base config)
    - `--force-fem` or `-ff` if present, will force fem stage even if not turned on in configs
    - `--force-model` or `-fm` if present, will force model stage even if not turned on in configs
- An example is: `python -m scripts.master_script --base-cfg configs/base/base_0.yaml -ff -fm`
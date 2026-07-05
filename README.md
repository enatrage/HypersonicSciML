# HypersonicSciML
Repository for the code used during my external collaboration to Prof. Süleyman Cengizci's TÜBİTAK 225M468 Project. In this project, we develop a SciML model to conduct post-processing and model training on hypersonic flows.

## Installation / Usage Guide
This project utilizes a hybrid environment provisioning strategy. The C++ physics backends (FEniCS, mshr) must be installed via Conda to avoid source-compilation failures, while the machine learning stack (PyTorch, WandB) is handled via Pip to ensure precise CUDA version matching. Additionally, dockerfiles are also included if one wants to use a simpler usage route.

### 1. Option: Manual Installation and Usage

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

**Usage:**
- Activate your venv for the project and CD into the root
- Login to wandb via `wandb login <your_api_key>`
- Run `python -m scripts.master_script` with the following add-ons:
    - `--base-cfg` or `-bc` followed by the base config path (this is required)
    - `--addon-cfg` or `-ac` followed by the addon config path (optional, will override entries in base config)
    - `--force-fem` or `-ff` if present, will force fem stage even if not turned on in configs
    - `--force-model` or `-fm` if present, will force model stage even if not turned on in configs
- An example is: `python -m scripts.master_script --base-cfg configs/base/base_0.yaml -ff -fm`

### 2. Option: Dockerfile 

**Step 1: WandB**

Add a `.env` file to the repository root, and specifically write the following to the file (make sure to save it as well):
```bash
# .env
WANDB_API_KEY=<your_40_character_wandb_api_key_here>
```

**Step 2: Running the Dockerfiles**

Make sure you are in the repository root. You can run the code using one of the following options, while aware of the operational notes.

#### 1. Option, FEM Only:

To only run the FEM stage using the Dockerfile, use the following command:
```bash
docker-compose run --rm fem-engine python scripts/master_script.py -bc <config path here, with the model flag off>
```

#### 2. Option, Model Only:

To only run the Model training stage using the Dockerfile, use the following command:
```bash
docker-compose run --rm model-engine python scripts/master_script.py -bc <config path here, with the fem flag off>
```

#### 3. Option, Monolith (Both):

To run both FEM and Model training at the same time using the Dockerfile (in an end-to-end monolith), use the following command:
```bash
docker-compose run --rm monolith python scripts/master_script.py -bc <config path here, flags to your liking>
```

#### Operational Notes:
- `--rm` flag: This ensures the container is immediately destroyed upon script exit (code 0 or crash), preventing filesystem clutter. Your data persists locally because of the volume mounts.

- Dependency Updates: If you modify `requirements.txt` or `environment.yml`, append the `--build` flag to the command to force Docker to reconstruct the pip/conda layers before executing.

- Detached Execution: For long-running training loops, if you want to detach your terminal session and let it run in the background, use up instead of run: `docker-compose up -d model-engine`.

- Master script specifics: Make sure to check the `Usage` section of `1. Option: Manual Installation and Usage` for the specific flags related to the additional run commands.

- The first run of each Dockerfile call will take long to be finished building, given that it would be building a cache. Every time the requirements get updated, you'd need to rebuild the docker image via adding a `--build` tag next to `--rm`. This will eventually result in stale images, which you can clean up via: `docker image prune -f`.

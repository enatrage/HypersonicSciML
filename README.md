# HypersonicSciML
Repository for the code used during my external collaboration to Prof. Süleyman Cengizci's TÜBİTAK 225M468 Project. In this project, we develop a SciML model to conduct post-processing and model training on hypersonic flows.

## User Guide ##
- Activate your venv for the project and CD into the root
- Login to wandb via `wandb login <your_api_key>`
- Run `python -m scripts.master_script.py` with the following add-ons:
    - `--base-cfg` or `-bc` followed by the base config path (this is required)
    - `--addon-cfg` or `-ac` followed by the addon config path (optional, will override entries in base config)
    - `--force-fem` or `-ff` if present, will force fem stage even if not turned on in configs
    - `--force-model` or `-fm` if present, will force model stage even if not turned on in configs
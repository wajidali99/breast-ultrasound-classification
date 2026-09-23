"""Shared helpers: seeding, device, config, and environment-aware paths.

Same code Kaggle, local machine, aur AWS SageMaker teeno par chale —
isi liye paths hardcode nahi kiye.
"""
import os
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def seed_everything(seed: int = 42, deterministic: bool = True) -> None:
    """Python, NumPy aur PyTorch (CPU + CUDA) sab ka seed fix karta hai."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """DataLoader workers ke liye (worker_init_fn=seed_worker)."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def detect_env() -> str:
    if os.environ.get("SM_CHANNEL_TRAIN"):
        return "sagemaker"
    if Path("/kaggle/input").exists():
        return "kaggle"
    return "local"


def get_paths() -> dict:
    """Environment ke hisaab se data aur output directories."""
    env = detect_env()
    if env == "sagemaker":
        data_root = Path(os.environ["SM_CHANNEL_TRAIN"])
        out_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    elif env == "kaggle":
        data_root = Path("/kaggle/input/breast-ultrasound-images-dataset/Dataset_BUSI_with_GT")
        out_dir = Path("/kaggle/working")
    else:
        data_root = Path(os.environ.get("BUSI_ROOT", "data/Dataset_BUSI_with_GT"))
        out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    return {"env": env, "data_root": data_root, "out_dir": out_dir}


def load_config(path: str, overrides: dict | None = None) -> dict:
    """YAML config load karta hai; overrides e.g. {"model.arch": "efficientnet_b0"}."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    for key, val in (overrides or {}).items():
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = val
    return cfg


def env_report() -> None:
    """Step 0 sanity check."""
    paths = get_paths()
    print(f"Environment : {paths['env']}")
    print(f"Data root   : {paths['data_root']}  (exists={paths['data_root'].exists()})")
    print(f"Output dir  : {paths['out_dir']}")
    print(f"PyTorch     : {torch.__version__}")
    print(f"CUDA        : {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}     : {torch.cuda.get_device_name(i)}")

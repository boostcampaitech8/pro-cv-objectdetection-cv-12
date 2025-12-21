import os
import random
import numpy as np
import torch
from mmengine.config import Config

def set_seed(seed=2025):
    """랜덤 시드 고정"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Random Seed set to {seed}")

def get_base_config(config_path, device="cuda", gpu_ids=[0]):
    """기본 Config 로드 및 디바이스 설정"""
    cfg = Config.fromfile(config_path)
    cfg.device = device
    cfg.gpu_ids = gpu_ids
    return cfg
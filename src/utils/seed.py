import torch
import numpy as np
import random
import os

def set_seed(seed: int = 42):
    """
    Forces all randomized operations to follow a deterministic path.
    Crucial for debugging and comparing model architectures.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Ensure deterministic behavior on hardware accelerators (MPS/CUDA)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
        
    print(f"[System] Random seed securely set to {seed}")
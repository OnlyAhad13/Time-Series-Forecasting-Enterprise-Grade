import random 
import numpy as np
import torch
import os


def set_seed(seed: int = 42, deterministic: bool = True):
    """
    Set random seeds for reproducibility

    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms(maybe slower but reproducible)
    """
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        #Set environment variable for deterministic operations
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        
        


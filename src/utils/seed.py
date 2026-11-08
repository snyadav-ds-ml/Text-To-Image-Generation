import random      # Python built-in randomness
import numpy as np  # numpy random ops
import torch        # PyTorch random ops


def set_seed(seed):
    random.seed(seed)                    # fixes Python random
    np.random.seed(seed)                 # fixes numpy random
    torch.manual_seed(seed)              # fixes CPU tensor ops
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed) # fixes GPU tensor ops

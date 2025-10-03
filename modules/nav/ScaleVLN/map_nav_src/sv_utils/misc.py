import random
import numpy as np
import torch

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)

# def length2mask(length, size=None):
#     batch_size = len(length)
#     size = int(max(length)) if size is None else size
#     mask = (torch.arange(size, dtype=torch.int64).unsqueeze(0).repeat(batch_size, 1)
#                 > (torch.LongTensor(length) - 1).unsqueeze(1)).cuda()
#     return mask

def length2mask(length, size=None):
    device = length.device if isinstance(length, torch.Tensor) else torch.device("cpu") # added 
    if isinstance(length, list):
        length = torch.tensor(length, dtype=torch.long)

    batch_size = len(length)
    size = int(max(length)) if size is None else size
    mask = (torch.arange(size, dtype=torch.int64, device=device).unsqueeze(0).repeat(batch_size, 1)
        > (length - 1).unsqueeze(1)) # modified
    return mask

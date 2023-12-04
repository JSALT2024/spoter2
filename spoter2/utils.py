import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed=0, device='cuda'):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_batch(data: torch.tensor):
    """
    data: [B, SEQ, DIM]
    """
    data = data.clone().detach().cpu().numpy()
    batch_size = data.shape[0]
    fig, ax = plt.subplots(1, batch_size, figsize=(5 * batch_size, 10))
    for i in range(batch_size):
        ax[i].imshow(data[i])
    plt.show()

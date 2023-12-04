import torch
import numpy as np


def collate_fn(batch: list, pad_token: torch.tensor):
    """
    batch: list([B, SEQ, DIM])
    """
    target_length = np.max([sample.shape[0] for sample in batch])

    _batch = []
    padding_idx = []
    for sample in batch:
        seq_len = sample.shape[0]
        pad_len = target_length - seq_len

        padding = pad_token.repeat((pad_len, 1))
        sample = torch.cat([sample, padding], dim=0)
        _batch.append(sample)
        padding_idx.append(seq_len)

    return {
        "data": torch.stack(_batch),
        "padding_idx": padding_idx
    }

import torch
import numpy as np


def collate_fn(batch: list):
    """
    batch: list([B, SEQ, DIM])
    """
    data = [d["data"] for d in batch]
    labels = [d["label"] for d in batch]

    pad_token = torch.zeros([1, data[0].shape[-1]])
    target_length = np.max([sample.shape[0] for sample in data])

    _batch = []
    padding_idx = []
    for sample in data:
        seq_len = sample.shape[0]
        pad_len = target_length - seq_len

        padding = pad_token.repeat((pad_len, 1))
        sample = torch.cat([sample, padding], dim=0)
        _batch.append(sample)
        padding_idx.append(seq_len)

    return {
        "data": torch.stack(_batch),
        "padding_idx": padding_idx,
        "labels": torch.tensor(labels, dtype=torch.long)
    }

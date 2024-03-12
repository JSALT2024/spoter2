import torch
import numpy as np


def collate_fn(batch: list):
    """
    batch: list([B, SEQ, DIM])
    """
    data = [d["data"] for d in batch]

    keys = list(batch[0].keys())
    keys.remove("data")
    other = {}
    for key in keys:
        other[key] = []
        for b in batch:
            value = b[key]
            other[key].append(value)
    for k, v in other.items():
        if isinstance(v[0], int):
            other[k] = torch.tensor(other[k], dtype=torch.long)

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

    new_batch = {
        "data": torch.stack(_batch),
        "padding_idx": padding_idx,
    }
    new_batch.update(other)

    return new_batch

import torch
import numpy as np


def mask_tokens(seq_len, mask_prob=0.1):
    idxs = np.arange(seq_len)
    np.random.shuffle(idxs)

    idx = np.ceil(len(idxs) * mask_prob).astype(int)
    mask_idxs = idxs[:idx]

    return mask_idxs


def collate_fn(batch: list, pad_token: torch.tensor, mask_prob: float = 0.1):
    # batch: list([seq, b, dim])
    target_length = np.max([sample.shape[0] for sample in batch])

    _batch = []
    batch_mask_idxs = []
    batch_padding_idxs = []

    for sample in batch:
        seq_len = sample.shape[0]
        pad_len = target_length - seq_len
        padding = pad_token.repeat((pad_len, 1))
        padding_idxs = seq_len + np.arange(pad_len)

        mask_idxs = mask_tokens(seq_len, mask_prob)
        sample = torch.cat([sample, padding], dim=0)

        _batch.append(sample)
        batch_mask_idxs.append(mask_idxs)
        batch_padding_idxs.append(padding_idxs)

    return {
        "data": torch.stack(_batch).permute(1, 0, 2),
        "mask_idxs": batch_mask_idxs,
        "padding_idxs": batch_padding_idxs,
    }

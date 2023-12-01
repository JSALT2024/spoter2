import torch
import numpy as np


def mask_tokens(sequence, mask_token, mask_prob=0.1):
    seq_len = sequence.shape[0]
    idxs = np.arange(seq_len)
    np.random.shuffle(idxs)

    idx = np.ceil(len(idxs) * mask_prob).astype(int)
    mask_idxs = idxs[:idx]
    target = []

    for idx in mask_idxs:
        target.append(torch.tensor(sequence[idx]))
        sequence[idx] = mask_token

    return sequence, mask_idxs, torch.stack(target)


def collate_fn(batch: list, pad_token: torch.tensor, msk_token: torch.tensor, mask_prob: float = 0.1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # batch: list([b, seq, dim])
    target_length = np.max([sample.shape[0] for sample in batch])
    _batch = []
    batch_mask_idxs = []
    targets = []
    for sample in batch:
        sample = sample.to(device)
        seq_len = sample.shape[0]
        pad_len = target_length - seq_len
        padding = pad_token.repeat((pad_len, 1))

        sample, mask_idxs, target = mask_tokens(sample, msk_token, mask_prob)
        sample = torch.cat([sample, padding], dim=0)
        _batch.append(sample)
        batch_mask_idxs.append(mask_idxs)
        targets.append(target)

    return {
        "data": torch.stack(_batch).permute(1, 0, 2),
        "mask_idxs": batch_mask_idxs,
        "target": targets
    }


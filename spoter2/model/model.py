import torch
from torch import nn
import numpy as np
from spoter2.model.positiona_encoding import LearnablePositionalEncoding


class SPOTEREncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int = 108,
                 max_frames: int = 256,
                 nhead: int = 6,
                 num_layers: int = 6,
                 pos_encoding: str = "learnable_uniform"
                 ):
        super().__init__()

        # define transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # define pos encoding
        self.pos_encoding = None
        if pos_encoding == "learnable_uniform":
            self.pos_encoding = LearnablePositionalEncoding(max_frames, hidden_dim)
        elif pos_encoding == "learnable_normal":
            self.pos_encoding = LearnablePositionalEncoding(max_frames, hidden_dim, 0.02)

        # define tokens
        self.mask_token = nn.Parameter(torch.rand(1, hidden_dim))
        self.pad_token = nn.Parameter(torch.rand(1, hidden_dim))

    def __initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=0.2)
        torch.nn.init.normal_(self.pad_token, std=0.2)

    def replace_padding(self, data: torch.tensor, padding_idx: list):
        batch_size, seq_len = data.shape[:2]
        for bi in range(batch_size):
            pad_len = seq_len - padding_idx[bi]
            if pad_len == 0:
                continue
            padding = self.pad_token.repeat((pad_len, 1))
            data[bi, padding_idx[bi]:, :] = padding
        return data

    def mask_input(self, data: torch.tensor, padding_idx: list, mask_ratio: float = 0.1):
        batch_size, seq_len = data.shape[:2]
        batch_mask_idxs = []
        batch_targets = []
        for bi in range(batch_size):
            mask_idxs = np.arange(padding_idx[bi])
            np.random.shuffle(mask_idxs)
            idx = np.ceil(len(mask_idxs) * mask_ratio).astype(int)
            mask_idxs = mask_idxs[:idx]

            target = data[bi][mask_idxs]
            data[bi][mask_idxs] = self.mask_token.repeat(len(mask_idxs), 1)

            batch_mask_idxs.append(mask_idxs)
            batch_targets.append(target)
        return data, batch_targets, batch_mask_idxs

    def forward(self, x: torch.tensor, padding_idx: list | None = None, mask_ratio: float = 0.1):
        """
        x: [B, SEQ, DIM]
        """
        # prepare input
        batch_size, seq_len = x.shape[:2]
        if padding_idx is None:
            padding_idx = [seq_len] * batch_size
        x = self.replace_padding(x, padding_idx)
        x, targets, mask_idxs = self.mask_input(x, padding_idx, mask_ratio)

        # add pos encoding
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)

        # apply transformer
        x = self.transformer_encoder(x)

        # get predictions
        predictions = []
        for bi in range(batch_size):
            prediction = x[bi][mask_idxs[bi]]
            predictions.append(prediction)

        return predictions, targets

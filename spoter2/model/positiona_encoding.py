import torch
from torch import nn


class LearnablePositionalEncoding(nn.Module):
    def __init__(self, max_length: int, dim: int, std: float = 0):
        super().__init__()

        parameters = torch.rand(1, max_length, dim)
        if std:
            parameters = parameters.normal_(mean=0, std=std)
        self.pos_encoding = nn.Parameter(parameters)

    def forward(self, x: torch.tensor):
        seq_len = x.shape[1]
        x = x + self.pos_encoding[:, :seq_len]
        return x

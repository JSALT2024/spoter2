import torch
from torch import nn
from spoter2.model.positiona_encoding import LearnablePositionalEncoding


class SPOTEREncoder(nn.Module):
    def __init__(self,
                 hidden_dim: int = 108,
                 max_frames: int = 256,
                 nhead: int = 9,
                 num_layers: int = 6,
                 pos_encoding: str = "learnable_uniform"
                 ):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pos_encoding = None
        if pos_encoding == "learnable_uniform":
            self.pos_encoding = LearnablePositionalEncoding(max_frames, hidden_dim)
        elif pos_encoding == "learnable_normal":
            self.pos_encoding = LearnablePositionalEncoding(max_frames, hidden_dim, 0.01)

        self.mask_token = nn.Parameter(torch.rand(1, hidden_dim))
        self.pad_token = nn.Parameter(torch.rand(1, hidden_dim))
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)

    def add_tokens(self, batch_data: dict):
        batch_size = batch_data["data"].shape[1]
        for b in range(batch_size):
            msk_idx = batch_data["mask_idxs"][b]
            pad_idx = batch_data["padding_idxs"][b]
            batch_data["data"][:, b, :][msk_idx] = self.mask_token
            batch_data["data"][:, b, :][pad_idx] = self.pad_token

    def forward(self, x):
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)  # expected input [seq, batch, feature]

        x = self.output_projection(x)
        return x

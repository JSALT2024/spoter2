import torch
from torch import nn
from spoter2.model.positiona_encoding import LearnablePositionalEncoding


class SPOTEREncoder(nn.Module):
    def __init__(self,
                 data_dim: int = 110,
                 hidden_dim: int = 256,
                 max_frames: int = 256,
                 nhead: int = 6,
                 num_layers: int = 6,
                 pos_encoding: str = "learnable_uniform"
                 ):
        super().__init__()

        # tokens
        self.pad_token = nn.Parameter(torch.rand(1, hidden_dim))

        # embedding layer
        self.input_embedding = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.GELU()
        )

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # pos encoding
        self.pos_encoding = None
        if pos_encoding == "learnable_uniform":
            self.pos_encoding = LearnablePositionalEncoding(max_frames, hidden_dim)
        elif pos_encoding == "learnable_normal":
            self.pos_encoding = LearnablePositionalEncoding(max_frames, hidden_dim, 0.02)
        elif "learnable_normal" in pos_encoding:
            _, std = pos_encoding.split("-")
            self.pos_encoding = LearnablePositionalEncoding(max_frames, hidden_dim, float(std))

    def _initialize_weights(self):
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

    def forward_embedding(self, x: torch.tensor, padding_idx: list | None = None):
        """
        x: [B, SEQ, DIM]
        """
        # input embedding
        x = self.input_embedding(x)

        # add padding
        batch_size, seq_len = x.shape[:2]
        if padding_idx is None:
            padding_idx = [seq_len] * batch_size
        x = self.replace_padding(x, padding_idx)

        return x

    def forward_encoding(self, x: torch.tensor):
        if self.pos_encoding is not None:
            x = self.pos_encoding(x)
        return x

    def forward(self, x: torch.tensor, padding_idx: list | None = None):
        """
        x: [B, SEQ, DIM]
        """
        # apply embedding and padding
        x = self.forward_embedding(x, padding_idx)

        # add pos encoding
        x = self.forward_encoding(x)

        # apply transformer
        x = self.transformer_encoder(x)

        return x


class SPOTERDecoder(nn.Module):
    pass


class SPOTER(nn.Module):
    pass

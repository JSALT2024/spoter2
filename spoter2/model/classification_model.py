import torch
from torch import nn
from spoter2.model import SPOTEREncoder
from collections import OrderedDict


class SPOTERDecoderClassification(nn.Module):
    def __init__(self,
                 hidden_dim: int = 256,
                 nhead: int = 6,
                 num_layers: int = 6,
                 num_classes: int = 300,
                 ):
        super().__init__()
        # tokens
        self.class_query = nn.Parameter(torch.rand(1, 1, hidden_dim))

        # classification layer
        self.output_projection = nn.Linear(hidden_dim, num_classes)

        # transformer encoder
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

    def forward(self, x: torch.tensor):
        """
        x: [B, SEQ, DIM]
        """
        batch_size = x.shape[0]
        query = self.class_query.repeat((batch_size, 1, 1))
        x = self.transformer_decoder(query, x)
        x = self.output_projection(x)
        x = x.squeeze(1)  # remove sequence dimension

        return x


class SPOTERClassification(nn.Module):
    def __init__(self,
                 data_dim: int = 110,
                 hidden_dim: int = 256,
                 max_frames: int = 256,
                 nhead: int = 6,
                 num_layers: int = 6,
                 pos_encoding: str = "learnable_uniform",
                 num_classes: int = 300
                 ):
        super().__init__()

        self.encoder = SPOTEREncoder(
            data_dim=data_dim,
            hidden_dim=hidden_dim,
            max_frames=max_frames,
            nhead=nhead,
            num_layers=num_layers,
            pos_encoding=pos_encoding
        )
        self.decoder = SPOTERDecoderClassification(
            hidden_dim=hidden_dim,
            nhead=nhead,
            num_layers=num_layers,
            num_classes=num_classes
        )

    @staticmethod
    def _replace_prefix(data: OrderedDict, old: str, new: str = ""):
        new_data = OrderedDict()
        for k, v in data.items():
            new_data[k.replace(old, new)] = v
        return new_data

    def load_encoder(self, path: str):
        model = torch.load(path)["model"]
        model = self._replace_prefix(model, "encoder.transformer_encoder", "transformer_encoder")
        model = self._replace_prefix(model, "encoder.pad_token", "pad_token")
        msg = self.encoder.load_state_dict(model, strict=False)
        print("missing_keys:", msg[0])
        print("unexpected_keys:", msg[1])

    def forward(self, x: torch.tensor, padding_idx: list | None = None):
        x = self.encoder(x, padding_idx)
        x = self.decoder(x)

        return x

import timeit
import torch
from torch import nn

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    runs = 1000

    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
    x = torch.ones(256, 16, 256).to(device)  # s b f
    t = timeit.timeit(lambda: transformer_encoder(x), number=runs)
    print(f"Data: [s, b, f], Transformer: batch_fits=False, Time: {t / runs:.3f}s")

    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
    x = torch.ones(16, 256, 256).to(device)  # b s f
    t = timeit.timeit(lambda: transformer_encoder(x), number=runs)
    print(f"Data: [b, s, f], Transformer: batch_fits=True, Time: {t / runs:.3f}s")

    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
    x = torch.ones(16, 256, 256).to(device)  # s b f
    timeit.timeit(lambda: transformer_encoder(x.permute(1, 0, 2)), number=runs)
    print(f"Data: [b, s, f] + permute(1,0,2), Transformer: batch_fits=False, Time: {t / runs:.3f}s")

import timeit
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_seq_first(number: int = 1000):
    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
    x = torch.ones(256, 16, 256).to(device)  # s b f
    t = timeit.timeit(lambda: transformer_encoder(x), number=number)
    print(f"Data: [s, b, f], Transformer: batch_fits=False, Time: {t / number:.3f}s")


def test_batch_first(number: int = 1000):
    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
    x = torch.ones(16, 256, 256).to(device)  # b s f
    t = timeit.timeit(lambda: transformer_encoder(x), number=number)
    print(f"Data: [b, s, f], Transformer: batch_fits=True, Time: {t / number:.3f}s")


def test_seq_first_permute(number: int = 1000):
    encoder_layer = nn.TransformerEncoderLayer(d_model=256, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6).to(device)
    x = torch.ones(16, 256, 256).to(device)  # s b f
    t = timeit.timeit(lambda: transformer_encoder(x.permute(1, 0, 2)), number=number)
    print(f"Data: [b, s, f] + permute(1,0,2), Transformer: batch_fits=False, Time: {t / number:.3f}s")


if __name__ == "__main__":

    runs = 100

    test_seq_first(runs)
    test_batch_first(runs)
    test_seq_first_permute(runs)

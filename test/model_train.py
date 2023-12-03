from functools import partial

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from spoter2.data import StructuredDummyDataset, collate_fn
from spoter2.model import SPOTEREncoder
from spoter2.utils import plot_batch
from spoter2.utils import set_seed


def test():
    epochs = 1500
    lr = 0.001
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    set_seed(0)
    model = SPOTEREncoder(108, 256, 9, 6, "learnable_uniform").to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 1e-3)

    dataset = StructuredDummyDataset(4, (128, 256), 108)
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=4,
        collate_fn=partial(
            collate_fn,
            pad_token=torch.zeros([1, 108])
        )
    )

    loss = []
    targets = []
    predictions = []

    data = next(iter(loader))
    plot_batch(data["data"])
    data["data"] = data["data"].to(device)

    model.train()
    for _ in tqdm(range(epochs)):
        data = next(iter(loader))
        data["data"] = data["data"].to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions, targets = model(data["data"], data["padding_idx"], 0.1)
        batch_loss = torch.mean(torch.stack([criterion(p, t) for p, t in zip(predictions, targets)]))

        batch_loss.backward()
        optimizer.step()
        scheduler.step()

        loss.append(batch_loss.item())

    fig, ax = plt.subplots(2, len(targets), figsize=(len(targets) * 5, 2 * 2))
    for i, (_t, _p) in enumerate(zip(targets, predictions)):
        ax[0, i].imshow(_t.detach().cpu())
        ax[1, i].imshow(_p.detach().cpu())
    ax[0, 0].set_ylabel("target")
    ax[1, 0].set_ylabel("prediction")
    plt.show()

    plt.plot(loss)
    plt.show()


if __name__ == "__main__":
    test()

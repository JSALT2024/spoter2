import torch
from functools import partial
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from spoter2.data import DummyDataset, StructuredDummyDataset, collate_fn


def test():
    dataset = StructuredDummyDataset()
    loader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=4,
        collate_fn=partial(
            collate_fn,
            pad_token=torch.zeros([1, 108])
        )
    )

    batch = next(iter(loader))
    data = batch["data"]
    print("data shape:", data.shape)
    print(batch["padding_idx"])

    fig, ax = plt.subplots(1, data.shape[0], figsize=(5 * data.shape[0], 10))
    for i, d in enumerate(data):
        ax[i].imshow(d)
    plt.show()


if __name__ == "__main__":
    test()

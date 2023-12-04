from functools import partial


import torch
from torch.utils.data import DataLoader

from spoter2.data import StructuredDummyDataset, collate_fn
from spoter2.model import SPOTEREncoder
from spoter2.utils import set_seed, plot_batch


def test():
    set_seed(0)
    model = SPOTEREncoder(108, 256, 6, 6, "")

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
    plot_batch(data)

    batch["data"] = model.replace_padding(batch["data"], batch["padding_idx"])
    plot_batch(data)

    batch["data"] = model.mask_input(batch["data"], batch["padding_idx"], 0.1)
    plot_batch(data)


if __name__ == "__main__":
    test()

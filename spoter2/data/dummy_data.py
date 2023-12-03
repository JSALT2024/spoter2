import torch
import numpy as np
from torch.utils.data import Dataset


class DummyDataset(Dataset):
    def __init__(self,
                 dataset_size: int = 1024,
                 sequence_size: tuple[int, int] = (32, 256),
                 features: int = 108
                 ):
        self.data = [np.random.rand(np.random.randint(*sequence_size), features) for _ in range(dataset_size)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.tensor(self.data[idx]).float()


class StructuredDummyDataset(DummyDataset):
    def __init__(self,
                 dataset_size: int = 1024,
                 sequence_size: tuple[int, int] = (32, 256),
                 features: int = 108
                 ):
        super().__init__()
        self.data = [self.generate_sample(sequence_size, features) for _ in range(dataset_size)]

    @staticmethod
    def generate_sequence(seq_len):
        phi = np.random.uniform(0, 360)
        f = np.random.uniform(0, 0.1)
        t = np.arange(seq_len)
        amp = np.random.uniform(0, 1)
        seq = amp * np.sin(2 * np.pi * f * t + phi)

        return seq

    def generate_sample(self, sequence_size=(128, 256), features=108):
        seq_len = np.random.randint(*sequence_size)
        sequence = np.array([self.generate_sequence(seq_len) for _ in range(features)]).T
        return sequence

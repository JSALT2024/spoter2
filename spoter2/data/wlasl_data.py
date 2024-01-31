from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np


class WLASLDataset(Dataset):
    def __init__(self, path: str, transforms=None):
        raw_data = pd.read_csv(path)
        self.data = []
        self.labels = []
        self.transforms = transforms

        keypoint_names = []
        for name in raw_data.columns:
            _name = name.split("_")
            if _name[-1] in ["X", "Y"]:
                keypoint_names.append("_".join(_name[:-1]))
        keypoint_names = set(keypoint_names)
        keypoint_names = np.sort(list(keypoint_names))

        for idx, row in raw_data.iterrows():
            keypoints = []
            for name in keypoint_names:
                x = [float(i) for i in row[f"{name}_X"].strip("[]").split(",")]
                y = [float(i) for i in row[f"{name}_Y"].strip("[]").split(",")]
                keypoints.extend([x, y])
            keypoints = np.array(keypoints).T

            self.labels.append(row.labels)
            self.data.append(keypoints)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        data = torch.tensor(data).float()

        label = self.labels[idx]
        torch.tensor(label, dtype=torch.long)

        if self.transforms:
            data = self.transforms(data)

        return {"data": data, "label": label}

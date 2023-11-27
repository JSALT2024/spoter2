from torch.utils.data import Dataset
import pandas as pd
import json

class How2SignDatasetPose(Dataset):
    """ Custom dataloader for how2sign dataset on pose features"""
    def __init__(self, json_path, video_path, transform=None):
        """
        Args:
            data_path (string): Path to the JSON file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = json_path
        self.video_path = video_path
        self.data = None
        self.targets = None
        self.load_data()
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of sample to be fetched.
        """
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

    def load_data(self):
        """
        Load data from JSON file
        """
        f = open(file_path, 'r', encoding='utf-8')
        data = json.load(f)
        f.close()
        joint_list = []
        for i in data['joints']:
            joint_list.append(data['joints'][i])


        self.data = [joint_list]
        self.targets = [data['SENTENCE']]

if __name__ == '__main__':
    file_path = '../datasets/1aJwX9nRlmk_2-2-rgb_front.json'
    data_val = How2SignDatasetPose(json_path=file_path, video_path=None)

    print(data_val.__getitem__(0))
    print(data_val.__len__())



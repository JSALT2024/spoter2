from torch.utils.data import Dataset
import pandas as pd

class How2SignDatasetAppearance(Dataset):
    """ Custom dataloader for how2sign dataset on apearence features"""
    def __init__(self, csv_path, video_path, transform=None):
        """
        Args:
            csv_path (string): Path to the csv file with annotations.
            video_path (string): Path to the video files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_path = csv_path
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
        x = self.video_path + self.data[index] + '.mp4'
        y = self.targets[index] #.encode('utf-8')

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

    def load_data(self):
        """
        Load data from csv file
        """
        df = pd.read_csv(self.data_path, sep='\t')
        self.data = df['SENTENCE_NAME']
        self.targets = df['SENTENCE']


if __name__ == '__main__':
    file_path = '../datasets/how2sign_realigned_val.csv'
    video_path = 'CESTA_K_VIDEU/'
    data_val = How2SignDatasetAppearance(csv_path=file_path, video_path=video_path)

    print(data_val.__getitem__(100))
    print(data_val.__len__())



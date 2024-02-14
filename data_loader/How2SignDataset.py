import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
import os
import cv2
import warnings
import matplotlib.pyplot as plt
import datetime
from operator import itemgetter

class How2SignDataset(Dataset):
    """ Custom dataset for how2sign dataset on pose features.
    args:
        json_pose_path (string): Path to dir with JSON annotations.
        video_file_path (string, optional): Path to dir with video files.
        transform (callable, optional): Optional transform to be applied
            on a sample.
    """
    def __init__(self, json_pose_path, video_file_path=None, transform=None):

        self.video_path = video_file_path
        self.json_pose_path = json_pose_path
        self.face_landmarks = [101, 214,  # left cheek top, bot
                               330, 434,  # right cheek top, bot
                               197, 195, 4, 1,  # nose rigid1, rigid2, flex, tip
                               295, 282, 283,  # right eyebrow
                               53, 52, 65,  # left eyebrow
                               263, 386, 362, 374,  # right eye
                               33, 159, 133, 145,  # left eye
                               40, 270, 91, 321,  # outer mouth sqruare
                               311, 81, 178, 402,  # inner mouth square
                               78, 308   #inner mouth corners
                               ]
        self.transform = transform

        self.data = self.load_data()

    def __getitem__(self, index):
        """
        args: index (int): Index
        returns:
            sample (dict): dict of pose features and metadata
        """
        if self.transform:
            return self.transform(self.data[index]['KPI'])
        else:
            return self.data[index]['KPI']

    def __len__(self):
        return len(self.data)

    def get_json_data(self):
        """
        Load data from json files that are in file self.json_pose_path.
        returns:
            json_data (list): list of json dict data
        """

        if not os.path.exists(self.json_pose_path):
            raise ValueError(f'Error: json_pose_path does not exist \n {self.json_pose_path}')
        if not os.path.isdir(self.json_pose_path):
            raise ValueError(f'Error> json_pose_path is not a directory \n {self.json_pose_path} ')

        json_paths = []
        for filename in os.listdir(self.json_pose_path):
            if filename.endswith(".json"):
                json_paths.append(os.path.join(self.json_pose_path, filename))

        json_data = []
        for file_path in json_paths:
            f = open(file_path, 'r', encoding='utf-8')
            json_data.append(json.load(f))
            f.close()

        return json_data

    def get_video_names(self):
        """
        Get video names for .mp4 videos in self.video_path dir.
        returns:
            video_paths (list): list of .mp4 files available in self.video_path
        """
        if self.video_path is None:
            return None

        if not os.path.exists(self.video_path):
            raise ValueError(f'Error: video_path does not exist \n {self.video_path}')
        if not os.path.isdir(self.video_path):
            raise ValueError(f'Error> video_path is not a directory \n {self.video_path} ')

        video_names = []
        for filename in os.listdir(self.video_path):
            if filename.endswith(".mp4"):
                video_names.append(filename.strip('.mp4'))

        return video_names

    def combine_data(self, json_data, video_names):
        """
        Insert video paths into json_data. If video_path is None, return json_data as is.
        If video_path does not contain video with name from json_data, delete entry from json_data.
        args:
            json_data (dict): dict of json data
            video_paths (list): list of video paths
        """

        if self.video_path is None:
            return json_data

        delete_idxs = []
        for idx, json_entry in enumerate(json_data):
            if json_entry['SENTENCE_NAME'] in video_names:
                json_data[idx]['VIDEO_PATH'] = os.path.join(self.video_path + json_entry['SENTENCE_NAME'] + '.mp4')
            else:
                warnings.warn(f'Warning: video_path does not contain video with name {json_entry["SENTENCE_NAME"]} \n SKIPING ENTRY.')
                delete_idxs.append(idx)
        for idx in sorted(delete_idxs, reverse=True):
            del json_data[idx]

        return json_data

    def load_data(self):

        video_names = self.get_video_names()
        json_data = self.get_json_data()

        combined_data = self.combine_data(json_data, video_names)

        data_out = []
        for json_entry in combined_data:
            data_entry = {'KPI': [],
                          'SENTENCE': json_entry['SENTENCE'],
                          'metadata': {'VIDEO_NAME': json_entry['VIDEO_NAME'],
                                       'SENTENCE_ID': json_entry['SENTENCE_NAME'],
                                       'START': json_entry['START'],
                                       'END': json_entry['END'],
                                       'VIDEO_PATH': json_entry['VIDEO_PATH'] if 'VIDEO_PATH' in json_entry else None},
                          'plot_metadata': {'POSE_LANDMARKS': [],
                                            'RIGHT_HAND_LANDMARKS': [],
                                            'LEFT_HAND_LANDMARKS': [],
                                            'FACE_LANDMARKS': []}
                          }

            for frame_id in json_entry['joints']:

                pose_vector = np.array(json_entry['joints'][frame_id]['pose_landmarks'])[:, 0:2]

                if len(json_entry['joints'][frame_id]['right_hand_landmarks']) == 0:
                    right_hand_vector = np.zeros((21, 2))
                else:
                    right_hand_vector = np.array(json_entry['joints'][frame_id]['right_hand_landmarks'])[:, 0:2]

                if len(json_entry['joints'][frame_id]['left_hand_landmarks']) == 0:
                    left_hand_vector = np.zeros((21, 2))
                else:
                    left_hand_vector = np.array(json_entry['joints'][frame_id]['left_hand_landmarks'])[:, 0:2]

                face_vector = np.array(itemgetter(*self.face_landmarks)(json_entry['joints'][frame_id]['face_landmarks']))[:, 0:2]

                data_entry['KPI'].append(np.concatenate((pose_vector.flatten(), right_hand_vector.flatten(), left_hand_vector.flatten(), face_vector.flatten()), axis=0))
                data_entry['plot_metadata']['POSE_LANDMARKS'].append(json_entry['joints'][frame_id]['pose_landmarks'])
                data_entry['plot_metadata']['RIGHT_HAND_LANDMARKS'].append(json_entry['joints'][frame_id]['right_hand_landmarks'])
                data_entry['plot_metadata']['LEFT_HAND_LANDMARKS'].append(json_entry['joints'][frame_id]['left_hand_landmarks'])
                data_entry['plot_metadata']['FACE_LANDMARKS'].append(json_entry['joints'][frame_id]['face_landmarks'])

            data_out.append(data_entry)
        return data_out

    def plot_points2video(self, index, video_name):
        if self.video_path is None:
            raise ValueError(f'Error: video_path is None, cannot plot. \n Aborting.')
        item = self.__getitem__(index)
        plot_metadata = item['plot_metadata']

        cap = cv2.VideoCapture(item['metadata']['VIDEO_PATH'])

        # Check if the video file opened successfully
        if not cap.isOpened():
            raise ValueError(f'Error: Couldnt open the video file. \n {video_path} \n Aborting.')

        ret, frame = cap.read()

        height, width, layers = frame.shape
        idx = 0
        video = cv2.VideoWriter(video_name, 0, 3, (width, height))

        while ret:
            frame = self.anotate_img(frame, plot_metadata, idx, (125, 255, 10))
            video.write(frame)
            ret, frame = cap.read()
            idx += 1

        cap.release()
        cv2.destroyAllWindows()
        video.release()

    def anotate_img(self, img, metadata, idx, color):
        for i in self.face_landmarks:
            img = cv2.circle(img, (int(metadata['FACE_LANDMARKS'][idx][i][0]),
                                   int(metadata['FACE_LANDMARKS'][idx][i][1])),
                             radius=0, color=color, thickness=-1)
        return img

    def plot_points(self, item_index=0):
        item = self.__getitem__(item_index)
        plot_metadata = item['plot_metadata']

        cap = cv2.VideoCapture(item['metadata']['VIDEO_PATH'])
        # Check if the video file opened successfully
        if not cap.isOpened():
            raise ValueError(f'Error: Couldnt open the video file. \n {video_path} \n Aborting.')

        ret, frame = cap.read()
        cap.release()

        for i in self.face_landmarks:
            plt.plot(plot_metadata['FACE_LANDMARKS'][0][i][0],
                     plot_metadata['FACE_LANDMARKS'][0][i][1],
                     marker='.', color="g", markersize=1)
        plt.imshow(frame[:, :, ::-1], interpolation='none')
        plt.show()

if __name__ == '__main__':
    prefix = 'all'

    json_path = '../datasets/'+prefix

    video_path = '/home/toofy/JSALT_videos/'

    start = datetime.datetime.now()
    data_val = How2SignDataset(json_pose_path=json_path,
                               video_file_path=video_path,
                               transform=None)

    print(datetime.datetime.now()-start)

    # data_val.plot_points2video(0, prefix+'_0.avi')
    # data_val.plot_points(1)
    # data_val.plot_points2video(1, prefix+'_1.avi')
    # data_val.plot_points2video(2, prefix+'_2.avi')
    # data_val.plot_points2video(3, prefix+'_3.avi')

    print(data_val.__len__())

    dataloader = DataLoader(data_val, batch_size=1, shuffle=True, num_workers=2)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, len(sample_batched))
        if i_batch == 4:
            break
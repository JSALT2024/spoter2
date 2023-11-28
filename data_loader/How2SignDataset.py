import logging
from torch.utils.data import Dataset
import pandas as pd
import json
import os
from itertools import chain
import cv2

class How2SignDataset(Dataset):
    """ Custom dataset for how2sign dataset on pose features.
    args:
        json_pose_path (string): Path to dir with JSON annotations.
        video_path (string): Path to dir with video files.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        use_appearance (bool): If True, the dataset will return video frames as well as pose features.
    """
    def __init__(self, json_pose_path, video_file_path, transform=None, use_appearance=True):

        self.video_path = video_file_path
        self.json_pose_path = json_pose_path
        self.use_appearance = use_appearance
        self.transform = transform

        if use_appearance:
            self.data = self.load_data_all(json_pose_path)
        else:
            self.data = self.load_data_json_only()

    def __getitem__(self, index):

        if self.transform:
            return self.transform(self.data[index])
        else:
            return self.data[index]

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
        Insert video paths into json_data.
        args:
            json_data (dict): dict of json data
            video_paths (list): list of video paths
        """
        for json_entry in json_data:
            if json_entry['VIDEO_NAME'] in video_names:
                json_entry['VIDEO_PATH'] = self.video_path + json_entry['VIDEO_NAME'] + '.mp4'
            else:
                print(f'Warning: video_path does not contain video with name {json_entry["VIDEO_NAME"]}')
                json_entry['VIDEO_PATH'] = None
        return None

    def load_data_all(self, json_pose_path):

        video_names = self.get_video_names()
        json_data = self.get_json_data()

        combined_data = self.combine_data(json_data, video_names)

        data_out = []
        for json_entry in combined_data:
            data_entry = {'KPI': [],
                          'FRAMES': [],
                          'SENTENCE': json_entry['SENTENCE'],
                          'metadata': {'VIDEO_ID': json_entry['VIDEO_ID'],
                                       'VIDEO_NAME': json_entry['VIDEO_NAME'],
                                       'SENTENCE_ID': json_entry['SENTENCE_ID'],
                                       'START_REALIGNED': json_entry['START_REALIGNED'],
                                       'END_REALIGNED': json_entry['END_REALIGNED'],
                                       'VIDEO_PATH': json_entry['VIDEO_PATH']}}

            cap = cv2.VideoCapture(json_entry['VIDEO_PATH'])

            # Check if the video file opened successfully
            if not cap.isOpened():
                print(f'Error: Couldnt open the video file. \n {video_path} \n Continuing with other videos.')
                continue

            for frame_id in json_entry['joints']:
                pose_vector = sum(json_entry['joints'][frame_id]['pose_landmarks'], []) + \
                              sum(json_entry['joints'][frame_id]['right_hand_landmarks'], []) + \
                              sum(json_entry['joints'][frame_id]['left_hand_landmarks'], [])
                # frame['face_landmarks']
                data_entry['KPI'].append(pose_vector)
                ret, frame = cap.read()
                if not ret:
                    print(f'Error: Couldnt read the frame. \n {video_path} \n Continuing with other videos.')
                    continue
                data_entry['FRAMES'].append(frame)

            data_out.append(data_entry)
        return data_out

    def load_data_json_only(self):

        json_data = self.get_json_data()

        data_out = []
        for json_entry in json_data:
            data_entry = {'KPI': [],
                          'SENTENCE': json_entry['SENTENCE'],
                          'metadata': {'VIDEO_ID': json_entry['VIDEO_ID'],
                                       'VIDEO_NAME': json_entry['VIDEO_NAME'],
                                       'SENTENCE_ID': json_entry['SENTENCE_ID'],
                                       'START_REALIGNED': json_entry['START_REALIGNED'],
                                       'END_REALIGNED': json_entry['END_REALIGNED'], }}
            for frame_id in json_entry['joints']:
                pose_vector = sum(json_entry['joints'][frame_id]['pose_landmarks'],[]) + \
                               sum(json_entry['joints'][frame_id]['right_hand_landmarks'],[]) + \
                               sum(json_entry['joints'][frame_id]['left_hand_landmarks'],[])
                #frame['face_landmarks']
                data_entry['KPI'].append(pose_vector)

            data_out.append(data_entry)
        return data_out

if __name__ == '__main__':
    json_path = '../datasets/'
    csv_path = '../datasets/how2sign_realigned_val.csv'
    video_path = '/home/toofy/JSALT_videos/'

    data_val = How2SignDataset(json_pose_path=json_path,
                               video_file_path=video_path,
                               transform=None,
                               use_appearance=True)

    print(data_val.__getitem__(0))
    print(data_val.__len__())



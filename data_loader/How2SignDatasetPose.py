import logging
from torch.utils.data import Dataset
import pandas as pd
import json
import os
from itertools import chain
import cv2

class How2SignDataset(Dataset):
    """ Custom dataloader for how2sign dataset on pose features"""
    def __init__(self, json_pose_path, csv_appearance_path, video_file_path, transform=None, use_appearance=True):

        self.video_path = video_file_path
        self.use_appearance = use_appearance
        if use_appearance:
            self.data = self.load_data_all(json_pose_path, csv_appearance_path)
        else:
            self.data = self.load_data_json_only(json_pose_path)
        self.transform = transform

    def __getitem__(self, index):

        #TODO return data based on use_appearance

        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)

    def load_data_all(self, json_pose_path, csv_appearance_path): #TODO refactor and break into methods of loading

        if not os.path.exists(json_pose_path):
            raise ValueError(f'Error: json_pose_path does not exist \n {json_pose_path}')
        if not os.path.isdir(json_pose_path):
            raise ValueError(f'Error> json_pose_path is not a directory \n {json_pose_path} ')

        if not os.path.exists(csv_appearance_path):
            raise ValueError(f'Error: csv_appearance_path does not exist \n {csv_appearance_path}')
        if not os.path.isfile(csv_appearance_path):
            raise ValueError(f'Error> csv_appearance_path is not a file \n {csv_appearance_path} ')

        df = pd.read_csv(csv_appearance_path, sep='\t')

        video_paths = []
        for filename in os.listdir(self.video_path):
            if filename.endswith(".mp4"):
                video_paths.append(os.path.join(self.video_path, filename))


        json_paths = []
        for filename in os.listdir(json_pose_path):
            if filename.endswith(".json"):
                json_paths.append(os.path.join(json_pose_path, filename))

        print(f'Loading {len(json_paths)} json files from {json_pose_path}')  # TODO later logging?

        json_data = []
        for file_path in json_paths:
            f = open(file_path, 'r', encoding='utf-8')
            json_data.append(json.load(f))
            f.close()

        data_out = []
        for json_entry, video_path in zip(json_data, video_paths):
            data_entry = {'KPI': [],
                          'FRAMES': [],
                          'SENTENCE': json_entry['SENTENCE'],
                          'metadata': {'VIDEO_ID': json_entry['VIDEO_ID'],
                                       'VIDEO_NAME': json_entry['VIDEO_NAME'],
                                       'SENTENCE_ID': json_entry['SENTENCE_ID'],
                                       'START_REALIGNED': json_entry['START_REALIGNED'],
                                       'END_REALIGNED': json_entry['END_REALIGNED'], }}

            cap = cv2.VideoCapture(video_path)

            # Check if the video file opened successfully
            if not cap.isOpened():
                raise ValueError(f'Error: Couldnt open the video file. \n {video_path} \n Continuing with other videos.')
                continue

            for frame_id in json_entry['joints']:
                pose_vector = sum(json_entry['joints'][frame_id]['pose_landmarks'], []) + \
                              sum(json_entry['joints'][frame_id]['right_hand_landmarks'], []) + \
                              sum(json_entry['joints'][frame_id]['left_hand_landmarks'], [])
                # frame['face_landmarks']
                data_entry['KPI'].append(pose_vector)
                ret, frame = cap.read()
                if not ret:
                    raise ValueError(f'Error: Couldnt read the frame. \n {video_path} \n Continuing with other videos.')
                    continue
                data_entry['FRAMES'].append(frame)

            data_out.append(data_entry)
        return data_out

    def load_data_json_only(self, json_pose_path):
        if not os.path.exists(json_pose_path):
            raise ValueError(f'Error: json_pose_path does not exist \n {json_pose_path}')
        if not os.path.isdir(json_pose_path):
            raise ValueError(f'Error> json_pose_path is not a directory \n {json_pose_path} ')

        json_paths = []
        for filename in os.listdir(json_pose_path):
            if filename.endswith(".json"):
                json_paths.append(os.path.join(json_pose_path, filename))

        print(f'Loading {len(json_paths)} json files from {json_pose_path}') #TODO later logging?

        json_data = []
        for file_path in json_paths:
            f = open(file_path, 'r', encoding='utf-8')
            json_data.append(json.load(f))
            f.close()

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
    json_path = '/media/toofy/kky_plzen4/projects/cv/JSALT/mediapipe/val_keypoints_meta'
    csv_path = '../datasets/how2sign_realigned_val.csv'
    video_path = '/media/toofy/kky_nas/JSALT/How2Sign/val_rgb_front_clips/raw_videos/'

    data_val = How2SignDataset(json_pose_path=json_path, csv_appearance_path=csv_path, video_file_path=video_path, transform=None, use_appearance=True)

    print(data_val.__getitem__(0))
    print(data_val.__len__())



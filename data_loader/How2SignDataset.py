from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import json
import os
import cv2
import warnings
import matplotlib.pyplot as plt

class How2SignDataset(Dataset):
    """ Custom dataset for how2sign dataset on pose features.
    args:
        json_pose_path (string): Path to dir with JSON annotations.
        video_path (string): Path to dir with video files.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        use_appearance (bool): If True, the dataset will return video frames as well as pose features.
    """
    def __init__(self, json_pose_path, video_file_path=None, transform=None, use_appearance=True): # TODO data2memory or live loading once visual is ready

        self.video_path = video_file_path
        self.json_pose_path = json_pose_path
        self.use_appearance = use_appearance
        self.transform = transform

        if use_appearance:
            self.data = self.load_data_all()
        else:
            self.data = self.load_data_json_only()

    def __getitem__(self, index):
        """
        args: index (int): Index
        returns:
            sample (dict): dict of pose features and metadata
        """
        if self.use_appearance:
            pass
        if self.transform:
            return self.transform(self.data[index]['KPI'])
        else:
            return self.data[index]['KPI']   #TODO tohle chce kuba na pretrain

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
        delete_idxs = []
        for idx, json_entry in enumerate(json_data):
            if json_entry['VIDEO_NAME'] in video_names:
                json_data[idx]['VIDEO_PATH'] = os.path.join(self.video_path + json_entry['VIDEO_NAME'] + '.mp4')
            else:
                warnings.warn(f'Warning: video_path does not contain video with name {json_entry["VIDEO_NAME"]} \n SKIPING ENTRY.')
                delete_idxs.append(idx)
        for idx in sorted(delete_idxs, reverse=True):
            del json_data[idx]

        return json_data

    def load_data_all(self):

        video_names = self.get_video_names()
        json_data = self.get_json_data()

        combined_data = self.combine_data(json_data, video_names)

        data_out = []
        for json_entry in combined_data:
            data_entry = {'KPI': [],
                          'FRAMES': [],
                          'SENTENCE': json_entry['SENTENCE'],
                          'metadata': {'VIDEO_NAME': json_entry['VIDEO_NAME'],
                                       'SENTENCE_ID': json_entry['SENTENCE_ID'],
                                       'START': json_entry['START_REALIGNED'],
                                       'END': json_entry['END_REALIGNED'],
                                       'VIDEO_PATH': json_entry['VIDEO_PATH']},
                          'plot_metadata': {'POSE_LANDMARKS': [],
                                            'RIGHT_HAND_LANDMARKS': [],
                                            'LEFT_HAND_LANDMARKS': [],
                                            'FACE_LANDMARKS': []}
                          }

            cap = cv2.VideoCapture(json_entry['VIDEO_PATH'])

            # Check if the video file opened successfully
            if not cap.isOpened():
                warnings.warn(f'Error: Couldnt open the video file. \n {video_path} \n Continuing with other videos.')
                continue

            # debug info
            # print(f" video len: {int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}")
            # print(f" json len: {len(json_entry['joints'])}")

            for frame_id in json_entry['joints']:
                pose_vector = sum(json_entry['joints'][frame_id]['pose_landmarks'], []) + \
                              sum(json_entry['joints'][frame_id]['right_hand_landmarks'], []) + \
                              sum(json_entry['joints'][frame_id]['left_hand_landmarks'], [])
                # frame['face_landmarks']
                data_entry['KPI'].append(pose_vector)
                data_entry['plot_metadata']['POSE_LANDMARKS'].append(json_entry['joints'][frame_id]['pose_landmarks'])
                data_entry['plot_metadata']['RIGHT_HAND_LANDMARKS'].append(json_entry['joints'][frame_id]['right_hand_landmarks'])
                data_entry['plot_metadata']['LEFT_HAND_LANDMARKS'].append(json_entry['joints'][frame_id]['left_hand_landmarks'])
                data_entry['plot_metadata']['FACE_LANDMARKS'].append(json_entry['joints'][frame_id]['face_landmarks'])

                ret, frame = cap.read()
                if not ret:
                    warnings.warn(f'Error: Couldn\'t read the frame. \n {video_path} \n Continuing with other videos.')
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
                          'metadata': {'VIDEO_NAME': json_entry['VIDEO_NAME'],
                                       'SENTENCE_ID': json_entry['SENTENCE_ID'],
                                       'START': json_entry['START_REALIGNED'],
                                       'END': json_entry['END_REALIGNED'], }}
            for frame_id in json_entry['joints']:
                pose_vector = sum(json_entry['joints'][frame_id]['pose_landmarks'],[]) + \
                               sum(json_entry['joints'][frame_id]['right_hand_landmarks'],[]) + \
                               sum(json_entry['joints'][frame_id]['left_hand_landmarks'],[])
                               #frame['face_landmarks'] #TODO face landmarks
                data_entry['KPI'].append(pose_vector)

            data_out.append(data_entry)
        return data_out

    def plot_points(self, index): #TODO finish once visual extraction is ready
        item = self.__getitem__(index)
        img = item['FRAMES'][5]
        plot_metadata = item['plot_metadata']

        inner_mouth = [78,191,80,81,82,13,312,311,310,415,308,324,318,402,317,14,87,178,88,95]
        middle_mouth = [11,302,303,304,408,292,307,320,404,315,16,85,180,90,77,76,184,74,73,72]
        outer_mouth = [0,267,269,270,409,291,375,321,405,314,17,84,181,91,146,61,185,40,39,37]

        nose = [1,4,5,195,197,6,168]
        left_eye = [159,145,33,133,153,144,159]
        right_eye = [386,374,263,362,382,373,386]

        left_eye_arch = [225, 224, 223, 222, 221]
        right_eye_arch = [445,444,443,442,441]

        left_eyebrow = [285,295,282,283,276,293,334,296,336]
        right_eyebrow = [55,107,66,105,63,46,53,52,65]

        forehead_all = [69,66,107,108,151,9,8,336,337,299,296]
        forehead_outline = [103,104,105,65,55,193,168,417,285,295,334,333,332,297,338,10,109,67]

        face_contour = [152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377]

        #TODO ASK
        # left_cheek = [101,123,137,93,132,58,172,135,214,207,205]
        # left_cheek_in = [50,147,187,192,213,177,215,138]
        # right_cheek = []

        #EYE PLOT
        for i in left_eye:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="g", markersize=1)

        for i in left_eye_arch:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="r", markersize=1)

        for i in left_eyebrow:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="b", markersize=1)

        for i in right_eye:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="g", markersize=1)

        for i in right_eye_arch:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="r", markersize=1)

        for i in right_eyebrow:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="b", markersize=1)
        plt.imshow(img, interpolation=None)
        plt.show(interpolation=None)


        # MOUTH PLOT
        for i in outer_mouth:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="r", markersize=1)

        for i in middle_mouth:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="g", markersize=1)

        for i in inner_mouth:
            plt.plot(plot_metadata['FACE_LANDMARKS'][5][i][0],
                     plot_metadata['FACE_LANDMARKS'][5][i][1],
                     marker='.', color="b", markersize=1)
        plt.imshow(img, interpolation=None)
        plt.show(interpolation=None)


if __name__ == '__main__':
    json_path = '../datasets/'
    video_path = '/home/toofy/JSALT_videos/'

    data_val = How2SignDataset(json_pose_path=json_path,
                               video_file_path=video_path,
                               transform=None,
                               use_appearance=True)

    data_val.plot_points(0)
    print(data_val.__len__())

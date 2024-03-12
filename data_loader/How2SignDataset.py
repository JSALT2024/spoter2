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

        self.json_paths = {}
        self.video_names = {}

        self.face_landmarks = [101, 214,  # left cheek top, bot
                               330, 434,  # right cheek top, bot
                               197, 195, 4, 1,  # nose rigid1, rigid2, flex, tip
                               295, 282, 283,  # right eyebrow
                               53, 52, 65,  # left eyebrow
                               263, 386, 362, 374,  # right eye
                               33, 159, 133, 145,  # left eye
                               40, 270, 91, 321,  # outer mouth sqruare
                               311, 81, 178, 402,  # inner mouth square
                               78, 308  # inner mouth corners
                               ]
        self.transform = transform

        self.get_json_names()
        self.get_video_names()

    def __getitem__(self, index):

        data = self.load_data(idx=index)
        if self.transform:
            return self.transform(data['KPI']), data['SENTENCE']
        else:
            return data['KPI'], data['SENTENCE']

    def __len__(self):
        return len(self.json_paths.keys())

    def get_json_names(self):
        """
        Load data from json files that are in file self.json_pose_path.
        returns:
            json_data (list): list of json dict data
        """

        if not os.path.exists(self.json_pose_path):
            raise ValueError(f'Error: json_pose_path does not exist \n {self.json_pose_path}')
        if not os.path.isdir(self.json_pose_path):
            raise ValueError(f'Error> json_pose_path is not a directory \n {self.json_pose_path} ')

        idx = 0

        for filename in os.listdir(self.json_pose_path):
            if filename.endswith(".json"):
                self.json_paths[idx] = os.path.join(self.json_pose_path, filename)
                idx += 1

    def get_video_names(self):
        """
        Get video names for .mp4 videos in self.video_path dir.
        returns:
            video_paths (list): list of .mp4 files available in self.video_path
        """
        if self.video_path is None:
            return

        if not os.path.exists(self.video_path):
            raise ValueError(f'Error: video_path does not exist \n {self.video_path}')
        if not os.path.isdir(self.video_path):
            raise ValueError(f'Error> video_path is not a directory \n {self.video_path} ')

        for filename in os.listdir(self.video_path):
            if filename.endswith(".mp4"):
                self.video_names[filename.strip('.mp4')] = None

    def load_data(self, idx=0):

        f = open(self.json_paths[idx], 'r', encoding='utf-8')
        json_data = json.load(f)
        f.close()

        if self.video_path is None:
            pass
        elif json_data['SENTENCE_NAME'] in self.video_names:
            json_data['VIDEO_PATH'] = os.path.join(self.video_path + json_data['SENTENCE_NAME'] + '.mp4')
        else:
            warnings.warn(
                f'Warning: video_path does not contain video with name {json_data["SENTENCE_NAME"]}')

        data_entry = {'KPI': [],
                      'SENTENCE': json_data['SENTENCE'],
                      'metadata': {'VIDEO_NAME': json_data['VIDEO_NAME'],
                                   'SENTENCE_ID': json_data['SENTENCE_NAME'],
                                   'START': json_data['START'],
                                   'END': json_data['END'],
                                   'VIDEO_PATH': json_data['VIDEO_PATH'] if 'VIDEO_PATH' in json_data else None},
                      'plot_metadata': {'POSE_LANDMARKS': [],
                                        'RIGHT_HAND_LANDMARKS': [],
                                        'LEFT_HAND_LANDMARKS': [],
                                        'FACE_LANDMARKS': []}
                      }

        kpi_mat = np.zeros((len(json_data['joints']), 214))
        for frame_id in json_data['joints']:

            pose_vector = np.array(json_data['joints'][frame_id]['pose_landmarks'])[:, 0:2]

            if len(json_data['joints'][frame_id]['right_hand_landmarks']) == 0:
                right_hand_vector = np.zeros((21, 2))
            else:
                right_hand_vector = np.array(json_data['joints'][frame_id]['right_hand_landmarks'])[:, 0:2]

            if len(json_data['joints'][frame_id]['left_hand_landmarks']) == 0:
                left_hand_vector = np.zeros((21, 2))
            else:
                left_hand_vector = np.array(json_data['joints'][frame_id]['left_hand_landmarks'])[:, 0:2]

            face_vector = np.array(itemgetter(*self.face_landmarks)(json_data['joints'][frame_id]['face_landmarks']))[:,0:2]

            kpi_mat[int(frame_id), :] = np.concatenate((pose_vector.flatten(), right_hand_vector.flatten(),
                                                        left_hand_vector.flatten(), face_vector.flatten()), axis=0)

            data_entry['plot_metadata']['POSE_LANDMARKS'].append(json_data['joints'][frame_id]['pose_landmarks'])
            data_entry['plot_metadata']['RIGHT_HAND_LANDMARKS'].append(json_data['joints'][frame_id]['right_hand_landmarks'])
            data_entry['plot_metadata']['LEFT_HAND_LANDMARKS'].append(json_data['joints'][frame_id]['left_hand_landmarks'])
            data_entry['plot_metadata']['FACE_LANDMARKS'].append(json_data['joints'][frame_id]['face_landmarks'])

        data_entry['KPI'] = kpi_mat
        return data_entry

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


class NoseNormalize(object):
    def __call__(self, sample):
        for index in range(0, len(sample)):
            kpi_x = sample[index, 0::2]
            kpi_y = sample[index, 1::2]
            x_max = np.max(kpi_x)
            y_max = np.max(kpi_y)
            x_min = np.min(kpi_x)
            y_min = np.min(kpi_y)

            kpi_x = (kpi_x - (x_max + x_min) / 2) / ((x_max - x_min) / 2)
            kpi_y = (kpi_y - (y_max + y_min) / 2) / ((y_max - y_min) / 2)

            sample[index, 0::2] = kpi_x
            sample[index, 1::2] = kpi_y

        return sample


def local_keypoint_normalization(joints: dict, landmarks: str, select_idx: list = [], padding: float = 0.1) -> dict:
    frames_keypoints = np.array([np.array(frame[landmarks])[:, :2] for frame in joints.values() if frame[landmarks]])
    frame_labels = [k for k, frame in joints.items() if frame[landmarks]]

    if select_idx:
        frames_keypoints = frames_keypoints[:, select_idx, :]

    # move to origin
    xmin = np.min(frames_keypoints[:, :, 0], axis=1)
    ymin = np.min(frames_keypoints[:, :, 1], axis=1)

    frames_keypoints[:, :, 0] -= xmin[:, np.newaxis]
    frames_keypoints[:, :, 1] -= ymin[:, np.newaxis]

    # pad to square
    xmax = np.max(frames_keypoints[:, :, 0], axis=1)
    ymax = np.max(frames_keypoints[:, :, 1], axis=1)

    dif_full = np.abs(xmax - ymax)
    dif = np.floor(dif_full / 2)

    for i in range(len(dif)):
        if xmax[i] > ymax[i]:
            ymax[i] += dif_full[i]
            frames_keypoints[i, :, 1] += dif[i]
        else:
            xmax[i] += dif_full[i]
            frames_keypoints[i, :, 0] += dif[i]

    # add padding to all sides
    side_size = np.max([xmax, ymax], axis=0)
    padding = side_size * padding

    frames_keypoints += padding[:, np.newaxis, np.newaxis]
    xmax += padding * 2
    ymax += padding * 2

    # normalize to [-1, 1]
    frames_keypoints /= xmax[:, np.newaxis, np.newaxis]
    frames_keypoints = frames_keypoints * 2 - 1

    frames_keypoints = dict(zip(frame_labels, frames_keypoints))
    return frames_keypoints


if __name__ == '__main__':
    prefix = 'person2'

    json_path = '../datasets/' + prefix

    video_path = '/home/toofy/JSALT_videos/'

    start = datetime.datetime.now()
    data_val = How2SignDataset(json_pose_path=json_path,
                               video_file_path=None,
                               transform=NoseNormalize())

    print(datetime.datetime.now() - start)

    # data_val.plot_points2video(0, prefix+'_0.avi')
    # data_val.plot_points(1)

    print(data_val.__len__())

    dataloader = DataLoader(data_val, batch_size=1, shuffle=True, num_workers=2)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, len(sample_batched))
        print(sample_batched[0].shape, sample_batched[1])
        if i_batch == 4:
            break

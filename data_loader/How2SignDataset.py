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
from typing import Tuple
import h5py


class How2SignDataset(Dataset):
    """ Custom dataset for how2sign dataset on pose features.
    args:
        h5_path (str): path to h5 file
        video_file_path (str, optional): path to video files
        transform (callable, optional): Optional transform to be applied on a sample.
    """

    def __init__(self, h5_path, video_file_path=None, transform=None):

        self.video_path = video_file_path
        self.h5_path = h5_path

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

        self.get_video_names()

    def __getitem__(self, index):

        data, sentence = self.load_data(idx=index)
        if self.transform:
            return self.transform(data), sentence
        else:
            return data, sentence

    def __len__(self):
        with h5py.File(self.h5_path, 'r') as f:
            return len(f.keys())

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
            raise ValueError(f'Error: video_path is not a directory \n {self.video_path} ')

        for filename in os.listdir(self.video_path):
            if filename.endswith(".mp4"):
                self.video_names[filename.strip('.mp4')] = os.path.join(self.video_path, filename)

    def load_data(self, idx=0):

        with h5py.File(self.h5_path, 'r') as f:
            video_name = list(f.keys())[idx]
            face_landmarks = f[video_name]['joints']['face_landmarks'][()]
            left_hand_landmarks = f[video_name]['joints']['left_hand_landmarks'][()]
            right_hand_landmarks = f[video_name]['joints']['right_hand_landmarks'][()]
            pose_landmarks = f[video_name]['joints']['pose_landmarks'][()]
            sentence = f[video_name]['sentence'][()].decode('utf-8')

        # if self.video_path is None:  #TODO implement once visual training is relevant
        #     pass
        # elif video_name in self.video_names:
        #     video_path = os.path.join(self.video_path + video_name + '.mp4')
        # else:
        #     warnings.warn(
        #         f'Warning: video_path does not contain video with name {video_name}')

        face_landmarks = face_landmarks[:, self.face_landmarks, 0:2]  # select only wanted KPI and  x, y
        left_hand_landmarks = left_hand_landmarks[:, :, 0:2]
        right_hand_landmarks = right_hand_landmarks[:, :, 0:2]
        pose_landmarks = pose_landmarks[:, :, 0:2]

        data = np.concatenate((pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks),
                              axis=1).reshape(len(face_landmarks), 214)

        return data, sentence

    def plot_points2video(self, index, video_name):
        if self.video_path is None:
            raise ValueError("Error: video_path is None, cannot plot. \n Aborting.")
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

class LocalKeypointNormalization(object):
    def __init__(self, landmarks):
        self.landmarks = landmarks

    def __call__(self, joints: dict, landmarks: str, select_idx: list = [], padding: float = 0.1) -> dict:
        frames_keypoints = np.array(
            [np.array(frame[landmarks])[:, :2] for frame in joints.values() if frame[landmarks]])
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

def get_keypoints(joints, landmarks_name):
    frames_keypoints = np.array(
        [np.array(frame[landmarks_name])[:, :2] for frame in joints.values() if frame[landmarks_name]])
    valid_frames = [i for i, frame in enumerate(joints.values()) if frame[landmarks_name]]

    _frames_keypoints = np.zeros([len(joints), *frames_keypoints.shape[1:]])
    _frames_keypoints[valid_frames] = frames_keypoints
    frames_keypoints = _frames_keypoints

    return frames_keypoints, valid_frames


def output_keypoints(joints, valid_frames, frames_keypoints):
    frames_names = np.array(list(joints.keys()))[valid_frames]
    frames_keypoints = frames_keypoints[valid_frames]

    return dict(zip(frames_names, frames_keypoints))


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


def global_keypoint_normalization(
        joints: dict,
        landmarks: str,
        add_landmarks_names: list,
        face_select_idx: list = [],
        sign_area_size: tuple = (1.5, 1.5),
        l_shoulder_idx: int = 11,
        r_shoulder_idx: int = 12) -> Tuple[dict, dict]:
    frames_keypoints, valid_frames = get_keypoints(joints, landmarks)

    # get distance between right and left shoulder
    l_shoulder_points = frames_keypoints[:, l_shoulder_idx, :]
    r_shoulder_points = frames_keypoints[:, r_shoulder_idx, :]
    distance = np.sqrt((l_shoulder_points[:, 0] - r_shoulder_points[:, 0]) ** 2 + (
                l_shoulder_points[:, 1] - r_shoulder_points[:, 1]) ** 2)

    # get center point between shoulders
    center_x = np.abs(l_shoulder_points[:, 0] - r_shoulder_points[:, 0]) / 2 + np.min(
        [l_shoulder_points[:, 0], r_shoulder_points[:, 0]], 0)
    center_y = np.abs(l_shoulder_points[:, 1] - r_shoulder_points[:, 1]) / 2 + np.min(
        [l_shoulder_points[:, 1], r_shoulder_points[:, 1]], 0)
    sign_area_size = np.array(sign_area_size) * distance[:, np.newaxis]

    # normalize
    frames_keypoints[:, :, 0] -= center_x[:, np.newaxis]
    frames_keypoints[:, :, 1] -= center_y[:, np.newaxis]

    frames_keypoints[:, :, 0] /= sign_area_size[:, 1, np.newaxis]
    frames_keypoints[:, :, 1] /= sign_area_size[:, 0, np.newaxis]

    # normalize additional landmarks
    add_landmarks = {}
    for add_landmarks_name in add_landmarks_names:
        add_frames_keypoints, add_valid_frames = get_keypoints(joints, add_landmarks_name)

        if face_select_idx and add_landmarks_name == "face_landmarks":
            add_frames_keypoints = add_frames_keypoints[:, face_select_idx, :]

        add_frames_keypoints[:, :, 0] -= center_x[:, np.newaxis]
        add_frames_keypoints[:, :, 1] -= center_y[:, np.newaxis]

        add_frames_keypoints[:, :, 0] /= sign_area_size[:, 1, np.newaxis]
        add_frames_keypoints[:, :, 1] /= sign_area_size[:, 0, np.newaxis]

        add_landmarks[add_landmarks_name] = output_keypoints(joints, add_valid_frames, add_frames_keypoints)

    frames_keypoints = output_keypoints(joints, valid_frames, frames_keypoints)

    return frames_keypoints, add_landmarks


if __name__ == '__main__':
    h5_path = '../datasets/H2S_val.h5'

    video_path = '/home/toofy/JSALT_videos/'

    start = datetime.datetime.now()
    data_val = How2SignDataset(h5_path=h5_path,
                               video_file_path=video_path,
                               transform=None)

    print(datetime.datetime.now() - start)

    print(data_val.__len__())

    dataloader = DataLoader(data_val, batch_size=1, shuffle=True, num_workers=3)
    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, len(sample_batched))
        print(sample_batched[0].shape, sample_batched[1])
        if i_batch == 4:
            break

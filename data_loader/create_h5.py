import os
import numpy as np
import h5py
import json
import argparse


def get_keypoints(json_data):
    right_hand_landmarks = []
    left_hand_landmarks = []
    face_landmarks = []
    pose_landmarks = []
    for frame_id in json_data['joints']:
        if len(json_data['joints'][frame_id]['pose_landmarks']) == 0:
            pose_landmarks.append(np.zeros((33, 4)))
        else:
            pose_landmarks.append(np.array(json_data['joints'][frame_id]['pose_landmarks']))

        if len(json_data['joints'][frame_id]['right_hand_landmarks']) == 0:
            right_hand_landmarks.append(np.zeros((21, 4)))
        else:
            right_hand_landmarks.append(np.array(json_data['joints'][frame_id]['right_hand_landmarks']))

        if len(json_data['joints'][frame_id]['left_hand_landmarks']) == 0:
            left_hand_landmarks.append(np.zeros((21, 4)))
        else:
            left_hand_landmarks.append(np.array(json_data['joints'][frame_id]['left_hand_landmarks']))

        if len(json_data['joints'][frame_id]['face_landmarks']) == 0:
            face_landmarks.append(np.zeros((478, 4)))
        else:
            face_landmarks.append(np.array(json_data['joints'][frame_id]['face_landmarks']))

    return pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks


def get_json_files(json_dir):
    json_files = [os.path.join(json_dir, json_file) for json_file in os.listdir(json_dir) if json_file.endswith('.json')]
    return json_files


def main():
    parser = argparse.ArgumentParser(description='Script for creating h5 file from json files.')
    parser.add_argument('--json_dir', help='Path to json dir', required=True)
    parser.add_argument('--h5_out', help='Path to save h5 in', required=True)
    args = vars(parser.parse_args())

    json_list = get_json_files(args['json_dir'])

    with h5py.File(args['h5_out'], 'w') as f:
        for json_file in json_list:

            with open(json_file, 'r') as file:
                keypoints_meta = json.load(file)

            video_name = keypoints_meta['SENTENCE_NAME']

            video_name_g = f.create_group(video_name)

            metadata = video_name_g.create_group('metadata')
            metadata.create_dataset(name='start_time', data=keypoints_meta['START'], dtype=np.float32)
            metadata.create_dataset(name='end_time', data=keypoints_meta['END'], dtype=np.float32)
            metadata.create_dataset(name='video_name', data=keypoints_meta['SENTENCE_NAME'])
            metadata.create_dataset(name='full_video_name', data=keypoints_meta['VIDEO_NAME'])

            video_name_g.create_dataset(name='sentence', data=keypoints_meta['SENTENCE'])

            joints_g = video_name_g.create_group('joints')

            pose_landmarks, right_hand_landmarks, left_hand_landmarks, face_landmarks = get_keypoints(keypoints_meta)

            joints_g.create_dataset(name='pose_landmarks', data=pose_landmarks, dtype=np.float32)
            joints_g.create_dataset(name='right_hand_landmarks', data=right_hand_landmarks, dtype=np.float32)
            joints_g.create_dataset(name='left_hand_landmarks', data=left_hand_landmarks, dtype=np.float32)
            joints_g.create_dataset(name='face_landmarks', data=face_landmarks, dtype=np.float32)

if __name__ == '__main__':
    main()

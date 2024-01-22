import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import json
import os
import matplotlib.pyplot as plt
import numpy as np

json_path = '../datasets/'
def get_json_data():
    json_paths = []
    for filename in os.listdir(json_path):
        if filename.endswith(".json"):
            json_paths.append(os.path.join(json_path, filename))

    json_data = []
    for file_path in json_paths:
        f = open(file_path, 'r', encoding='utf-8')
        json_data.append(json.load(f))
        f.close()

    return json_data

def center_data(data):
    data_out = []
    for d in data:
        nose_x = d[2] #0/1 je x/y koord rtu
        nose_y = d[3]
        data_centered = []
        for idx, val in enumerate(d, start=1):
            if idx % 2 == 0:
                data_centered.append(val-nose_y)
            else:
                data_centered.append(val-nose_x)
        data_out.append(data_centered)
    return data_out


json_data = get_json_data()

dataset = []
for json_entry in json_data:

    for frame_id in json_entry['joints']:
        pose_vector = []

        for i in json_entry['joints'][frame_id]['face_landmarks']:
            pose_vector.extend([i[0]])
            pose_vector.extend([i[1]])
        dataset.append(pose_vector)


data = pd.DataFrame(dataset)

sc = StandardScaler()

scaled_data_nose = center_data(dataset)
print(f"Scaled data with nose: {scaled_data_nose[0]}")
centered_data = pd.DataFrame(scaled_data_nose)

scaled_data = sc.fit_transform(centered_data)

pca = PCA()

pca_data = pca.fit_transform(scaled_data)
explained_variance = pca.explained_variance_ratio_
eigen_vectors = pca.components_

print(f"scaled variance {explained_variance}")
print(pca.components_)

cumulative = np.cumsum(explained_variance)
plt.plot(cumulative)
plt.show()

pca_data = pca.fit_transform(centered_data)
explained_variance = pca.explained_variance_ratio_
print(f"centered variance {explained_variance}")

cumulative = np.cumsum(explained_variance)
plt.plot(cumulative)
plt.show()
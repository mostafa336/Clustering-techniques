import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


k_values = [8, 13, 19, 28, 38]
def load_data(folder_path):
    data = []
    labels = []
    for activity_folder in os.listdir(folder_path):
        activity_path = os.path.join(folder_path, activity_folder)
        for subject_folder in os.listdir(activity_path):
            subject_path = os.path.join(activity_path, subject_folder)
            for segment_file in os.listdir(subject_path):
                segment_path = os.path.join(subject_path, segment_file)
                segment_data = np.loadtxt(segment_path, delimiter=',')
                data.append(segment_data)
                labels.append(int(activity_folder[1:]))  # Extract activity number
    return np.array(data), np.array(labels)

folder_path = "data"
data, labels = load_data(folder_path)
print("Data shape:", data.shape)
print("Labels shape:", labels.shape)


# Solution 1: Taking the mean of each column in each segment
solution1_data = np.mean(data, axis=1)
print("Solution 1 data shape:", solution1_data.shape)

# Solution 2: Flattening all the features together
solution2_data = data.reshape(data.shape[0], -1)
print("Solution 2 data shape:", solution2_data.shape)

# Scaling the data
scaler = StandardScaler()
solution1_data_scaled = scaler.fit_transform(solution1_data)
print("Solution 1 data scaled shape:", solution1_data_scaled.shape)
solution2_data_scaled = scaler.fit_transform(solution2_data)
print("Solution 2 data scaled shape:", solution2_data_scaled.shape)
# Dimensionality reduction for Solution 2 using PCA
pca = PCA(n_components=100)
solution2_data_pca = pca.fit_transform(solution2_data_scaled)
print("Solution 2 data PCA shape:", solution2_data_pca.shape)


def load_data_split(folder_path, segments_per_subject=60, segments_for_training=48):
    train_data = []
    train_labels = []
    eval_data = []
    eval_labels = []
    for activity_folder in os.listdir(folder_path):
        activity_path = os.path.join(folder_path, activity_folder)
        for subject_folder in os.listdir(activity_path):
            subject_path = os.path.join(activity_path, subject_folder)
            subject_data = []
            for segment_file in os.listdir(subject_path):
                segment_path = os.path.join(subject_path, segment_file)
                segment_data = np.loadtxt(segment_path, delimiter=',')
                subject_data.append(segment_data)
            subject_data = np.array(subject_data)
            # Split data for training and evaluation
            train_data.extend(subject_data[:segments_for_training])
            eval_data.extend(subject_data[segments_for_training:])
            train_labels.extend([int(activity_folder[1:])] * segments_for_training)
            eval_labels.extend([int(activity_folder[1:])] * (segments_per_subject - segments_for_training))
    return np.array(train_data), np.array(train_labels), np.array(eval_data), np.array(eval_labels)

# Load data for training and evaluation
train_data, train_labels, eval_data, eval_labels = load_data_split(folder_path)


#
# # Solution 1: Taking the mean of each column in each segment
# solution1_train_data = np.mean(train_data, axis=1)
# print("Solution 1 train data shape:", solution1_train_data.shape)
#
# # Solution 2: Flattening all the features together
# Error
# solution2_train_data = data.reshape(train_data.shape[0], -1)
# print("Solution 2 train data shape:", solution2_train_data.shape)
#
# solution1_eval_data = np.mean(eval_data, axis=1)
# print("Solution 1 eval data shape:", solution1_eval_data.shape)
#
# solution2_eval_data = data.reshape(eval_data.shape[0], -1)
# print("Solution 2 eval data shape:", solution2_eval_data.shape)
import math
import os
import sys

import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

k_values = [8, 13, 19, 28, 38]
EVAL_DATA_SIZE = 1824


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


# data, labels = load_data(folder_path)
# print("Data shape:", data.shape)
# print("Labels shape:", labels.shape)
#
#
# # Solution 1: Taking the mean of each column in each segment
# solution1_data = np.mean(data, axis=1)
# print("Solution 1 data shape:", solution1_data.shape)
#
# # Solution 2: Flattening all the features together
# solution2_data = data.reshape(data.shape[0], -1)
# print("Solution 2 data shape:", solution2_data.shape)
#
# # Scaling the data
# scaler = StandardScaler()
# solution1_data_scaled = scaler.fit_transform(solution1_data)
# print("Solution 1 data scaled shape:", solution1_data_scaled.shape)
# solution2_data_scaled = scaler.fit_transform(solution2_data)
# print("Solution 2 data scaled shape:", solution2_data_scaled.shape)
# # Dimensionality reduction for Solution 2 using PCA
# pca = PCA(n_components=100)
# solution2_data_pca = pca.fit_transform(solution2_data_scaled)
# print("Solution 2 data PCA shape:", solution2_data_pca.shape)


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


def extract_ground_truth(eval_data, partition_size=96):
    num_partitions = len(eval_data) // partition_size
    ground_truth_partitions = []

    for i in range(num_partitions):
        start_idx = i * partition_size
        end_idx = (i + 1) * partition_size
        partition = eval_data[start_idx:end_idx]
        ground_truth_partitions.append(partition)

    return ground_truth_partitions


def Entropy(cluster, ground_truth):
    N = len(cluster)
    entropy = 0
    for partition in ground_truth:
        partition_set = set(map(tuple, partition))
        intersect = len(cluster.intersection(partition_set))
        if intersect != 0:
            entropy -= (intersect / N) * (math.log2(intersect / N))
    return entropy


def calculate_entropy(clusters, ground_truth):
    entropy = 0
    for cluster in clusters:
        cluster_set = set(map(tuple, cluster))
        entropy += (len(cluster) / EVAL_DATA_SIZE) * Entropy(cluster_set, ground_truth)
    print("Entropy =", entropy)


def Purity(cluster, ground_truth):
    cluster_size = len(cluster)
    max_intersection = 0

    for partition in ground_truth:
        partition_set = set(map(tuple, partition))
        intersect = len(cluster.intersection(partition_set))
        max_intersection = max(intersect, max_intersection)

    return max_intersection / cluster_size


def Rec(cluster, ground_truth):
    max_intersection = 0
    size = 0
    for partition in ground_truth:
        partition_set = set(map(tuple, partition))
        intersect = len(cluster.intersection(partition_set))
        if intersect > max_intersection:
            size = len(partition)
            max_intersection = intersect
    return max_intersection / size


def calculate_purity(clusters, ground_truth):
    purity = 0
    for cluster in clusters:
        cluster_set = set(map(tuple, cluster))
        purity += (len(cluster) / EVAL_DATA_SIZE) * Purity(cluster_set, ground_truth)
    print("Purity =", purity)


def calculate_F_measure(clusters, ground_truth):
    f = 0
    for cluster in clusters:
        cluster_set = set(map(tuple, cluster))
        precision = Purity(cluster_set, ground_truth)
        recall = Rec(cluster_set, ground_truth)
        f += (2 * precision * recall) / (precision + recall)
    f = f / len(clusters)
    print("F-measure = ", f)


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Function to initialize centroids randomly
def initialize_centroids(data, k):
    centroids = np.empty((k, data.shape[1]))
    for i in range(k):
        # Select random data point as initial centroid
        centroid_idx = np.random.randint(0, len(data))
        centroids[i] = data[centroid_idx]
    return centroids


# K-Means clustering algorithm implementation
# K-Means clustering algorithm implementation
def kmeans(data, k):
    # Initialize centroids
    centroids = initialize_centroids(data, k)
    it = 1
    while True:  # Continue until convergence
        print("Iteration:", it)
        it += 1
        # Assign data points to closest centroids
        clusters = []
        for point in data:
            distances = np.array([euclidean_distance(point, centroid) for centroid in centroids])
            cluster_idx = np.argmin(distances)
            clusters.append(cluster_idx)

        # Convert clusters list to numpy array
        clusters = np.array(clusters)

        # Compute new centroids
        new_centroids = np.empty((k, data.shape[1]))
        for i in range(k):
            cluster_data = data[clusters == i]
            if len(cluster_data) > 0:  # Avoid division by zero
                new_centroids[i] = np.mean(cluster_data, axis=0)
            else:
                # If no data points assigned to the cluster, set centroid to None
                new_centroids[i] = None

        # Check convergence by comparing centroids
        if np.all(centroids == new_centroids):
            break  # Break if centroids have not changed

        # Update centroids
        centroids = new_centroids.copy()

    return centroids, clusters


def assign_to_nearest_centroid(data, centroids):
    labels = []
    for point in data:
        distances = np.array([euclidean_distance(point, centroid) for centroid in centroids])
        nearest_centroid_idx = np.argmin(distances)
        labels.append(nearest_centroid_idx)

    clusters = [[] for _ in range(centroids.shape[0])]
    for i, cluster_idx in enumerate(labels):
        clusters[cluster_idx].append(data[i])
    return np.array(labels), clusters


# Load data for training and evaluation
train_data, train_labels, eval_data, eval_labels = load_data_split(folder_path)
scaler = StandardScaler()

# Solution 1: Taking the mean of each column in each segment
solution1_train_data = np.mean(train_data, axis=1)
# print("Solution 1 train data shape:", solution1_train_data.shape)

solution1_eval_data = np.mean(eval_data, axis=1)
# print("Solution 1 eval data shape:", solution1_eval_data.shape)

# Solution 2: Flattening all the features together
# Reshape the training data for Solution 2
# solution2_train_data = train_data.reshape(train_data.shape[0], -1)
# print("Solution 2 train data shape:", solution2_train_data.shape)
#
# solution2_eval_data = eval_data.reshape(eval_data.shape[0], -1)
# print("Solution 2 eval data shape:", solution2_eval_data.shape)

solution1_train_data_scaled = scaler.fit_transform(solution1_train_data)
# print("Solution 1 train data scaled shape:", solution1_train_data_scaled.shape)
# solution2_train_data_scaled = scaler.fit_transform(solution2_train_data)
# print("Solution 2 train data scaled shape:", solution2_train_data_scaled.shape)
solution1_eval_data_scaled = scaler.fit_transform(solution1_eval_data)
# print("Solution 1 eval data scaled shape:", solution1_eval_data_scaled.shape)

solution1_ground_truth = extract_ground_truth(solution1_eval_data_scaled)
# print("Ground truth shape:", len(solution1_ground_truth), solution1_ground_truth[0].shape)
# solution2_eval_data_scaled = scaler.fit_transform(solution2_eval_data)
# print("Solution 2 eval data scaled shape:", solution2_eval_data_scaled.shape)

# Dimensionality reduction for Solution 2 using PCA
# pca = PCA(n_components=100)
# solution2_train_data_pca = pca.fit_transform(solution2_train_data_scaled)
# print("Solution 2 train data PCA shape:", solution2_train_data_pca.shape)
# solution2_eval_data_pca = pca.transform(solution2_eval_data_scaled)
# print("Solution 2 eval data PCA shape:", solution2_eval_data_pca.shape)


# K-means clustering
# Function to calculate Euclidean distance


# K-means clustering for Solution 1
k = 19
centroids, clusters2 = kmeans(solution1_train_data_scaled, k)
# print("Centroids shape:", centroids.shape)
# print("Clusters shape:", len(clusters2))
# print("Clusters:", clusters2)
# print("Centroids:", centroids)
predicted_labels, clusters = assign_to_nearest_centroid(solution1_eval_data_scaled, centroids)
# print("Predicted labels shape:", predicted_labels.shape)
# print("Predicted labels:", predicted_labels)

# Calculate purity
calculate_purity(clusters, solution1_ground_truth)
calculate_F_measure(clusters, solution1_ground_truth)
calculate_entropy(clusters, solution1_ground_truth)
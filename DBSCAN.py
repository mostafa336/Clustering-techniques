import math
import os
import numpy as np
from sklearn.cluster import KMeans, DBSCAN, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, confusion_matrix
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler
from collections import Counter



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


def euclidean_distance(point1, point2):
    return np.linalg.norm(point1 - point2)


def find_neighbors(data, point_index, eps):
    neighbors = []
    for i, other_point in enumerate(data):
        if i != point_index and euclidean_distance(data[point_index], other_point) <= eps:
            neighbors.append(i)
    return neighbors


def dbscan(data, eps, min_samples):
    num_points = len(data)
    visited = [False] * num_points
    cluster_labels = [-1] * num_points  # Initialize all points as noise (-1)

    cluster_id = 0
    for i in range(num_points):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = find_neighbors(data, i, eps)

        if len(neighbors) < min_samples:
            cluster_labels[i] = -1
        else:
            cluster_id += 1
            cluster_labels[i] = cluster_id
            expand_cluster(data, i, neighbors, cluster_id, eps, min_samples, visited, cluster_labels)

    return cluster_labels


def expand_cluster(data, point_index, neighbors, cluster_id, eps, min_samples, visited, cluster_labels):
    for neighbor in neighbors:
        if not visited[neighbor]:
            visited[neighbor] = True
            neighbor_neighbors = find_neighbors(data, neighbor, eps)
            if len(neighbor_neighbors) >= min_samples:
                neighbors.extend(neighbor_neighbors)

        if cluster_labels[neighbor] == -1:
            cluster_labels[neighbor] = cluster_id

def calculate_cluster_purity(cluster_labels, cluster, true_labels):
    cluster_indices = np.where(cluster_labels == cluster)[0]
    true_labels_in_cluster = true_labels[cluster_indices]
    cluster_size = len(cluster_indices)
    label_counts = Counter(true_labels_in_cluster)
    majority_label_count = max(label_counts.values())
    return majority_label_count

def calculate_purity(cluster_labels, true_labels):
    clusters = np.unique(cluster_labels)
    total_samples = len(cluster_labels)
    total_purity = 0

    for cluster in clusters:
        if cluster == -1:  # Skip noise points
            continue

        cluster_purity = calculate_cluster_purity(cluster_labels, cluster, true_labels)
        total_purity += cluster_purity


    overall_purity = total_purity / total_samples
    return overall_purity


def cond_entropy(cluster_labels, true_labels):
    cond_entropy = 0
    for cluster in np.unique(cluster_labels):
        cluster_indices = np.where(cluster == cluster_labels)[0]
        true_labels_in_cluster = true_labels[cluster_indices]
        temp = np.zeros(19)
        for i in range(0, len(cluster_indices)):
            temp[true_labels_in_cluster[i] - 1] += 1

        temp = temp / len(cluster_indices)
        temp2 = 0

        for i in range(19):
            if temp[i] != 0:
                temp2 -= temp[i] * math.log2(temp[i])

        cond_entropy += (temp2 * (len(cluster_indices) / len(true_labels)))

    return cond_entropy


def calculate_recall(cluster_labels, true_labels):
    clusters = np.unique(cluster_labels)
    total_true_positive = 0
    total_false_negative = 0

    # compute false negative
    num_labels = np.zeros((len(clusters), 19))
    for i, cluster in enumerate(clusters):
        if cluster == -1:
            continue
        cluster_indices = np.where(cluster_labels == cluster)[0]
        true_labels_in_cluster = true_labels[cluster_indices]
        for label in true_labels_in_cluster:
            if label < 1:
                continue
            num_labels[i][label - 1] += 1

    # loop for each label
    for j in range(0, 19):
        # loop for each cluster
        for i in range(0, len(clusters)):
            # check clusters below it
            for k in range(i + 1, len(clusters)):
                total_false_negative = num_labels[i][j] * num_labels[k][j]

    for cluster in clusters:
        if cluster == -1:  # Skip noise points
            continue

        cluster_indices = np.where(cluster_labels == cluster)[0]
        true_labels_in_cluster = true_labels[cluster_indices]
        label_counts = Counter(true_labels_in_cluster)

        # Calculate true positives
        if len(label_counts) > 0:
            majority_label = max(label_counts, key=label_counts.get)
            true_positive = label_counts[majority_label]
            total_true_positive += true_positive

    recall = total_true_positive / (total_true_positive + total_false_negative)

    return recall, num_labels

def calculate_F_measure(cluster_labels, true_labels):
    f_measure_sum = 0
    _, number_labels = calculate_recall(cluster_labels, true_labels)
    sum_of_labels = np.sum(number_labels, axis=0)
    print("sum of labels : ", sum_of_labels)
    for i, cluster_label in enumerate(np.unique(cluster_labels)):
        # precision
        precision = calculate_cluster_purity(cluster_labels, cluster_label, true_labels)
        # recall
        cluster_indices = np.where(cluster_labels == cluster_label)[0]
        true_labels_in_cluster = true_labels[cluster_indices]
        label_counts = Counter(true_labels_in_cluster)
        majority_label = max(label_counts, key=label_counts.get)
        cnt_major = label_counts[majority_label-1]
        if sum_of_labels[majority_label-1] == 0:
            continue
        recall = cnt_major / sum_of_labels[majority_label-1]

        if precision + recall != 0:
            f_measure = (2 * precision * recall) / (precision + recall)
            f_measure_sum += f_measure
    f_measure_avg = f_measure_sum / len(np.unique(cluster_labels))
    return f_measure_avg


folder_path = "data"
train_data, train_labels, eval_data, eval_labels = load_data_split(folder_path)
# solution 1
solution1_train_data = np.mean(train_data, axis=1)
solution1_eval_data = np.mean(eval_data, axis=1)
scaler = StandardScaler()
solution1_train_data_scaled = scaler.fit_transform(solution1_train_data)
solution1_eval_data_scaled = scaler.fit_transform(solution1_eval_data)

eps = 2
min_samples = 1
# evaluation of solution1 train
# train_cluster_labels = dbscan(solution1_train_data_scaled, eps, min_samples)
# purity = calculate_purity(train_cluster_labels, train_labels)
# recall = calculate_recall(train_cluster_labels, train_labels)
# cond = cond_entropy(train_cluster_labels, train_labels)
# print("train sol 1\npurity: ", purity, "\nrecall", recall, "\ncond", cond)

# evaluation data solution1
test_cluster_labels = dbscan(solution1_eval_data_scaled, eps, min_samples)
print("NUmber of Clusters : ", len(np.unique(test_cluster_labels)))
# purity1 = calculate_purity(test_cluster_labels, eval_labels)
recall1, _ = calculate_recall(test_cluster_labels, eval_labels)
print("recall : ", recall1)
F1 = calculate_F_measure(test_cluster_labels, eval_labels)
print("test f measure : ", F1)
# cond1 = cond_entropy(test_cluster_labels, eval_labels)
# print("test sol 1\npurity: ", purity1, "\nrecall: ", recall1, "\ncond", cond1)

# # Solution 2 train data
# solution2_train_data = train_data.reshape(train_data.shape[0], -1)
# solution2_data_scaled = scaler.fit_transform(solution2_train_data)
# pca = PCA(n_components=100)
# solution2_data_pca = pca.fit_transform(solution2_data_scaled)
# train2_cluster_labels = dbscan(solution2_data_pca, eps, min_samples)


# # solution 2 test data
# solution2_test_data = eval_data.reshape(eval_data.shape[0], -1)
# solution2_test_data_scaled = scaler.fit_transform(solution2_test_data)
# pca = PCA(n_components=100)
# solution2_test_data_pca = pca.fit_transform(solution2_test_data_scaled)
# test2_cluster_labels = dbscan(solution2_test_data_pca, eps, min_samples)
import math
import os
from collections import Counter

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler


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


def calculate_cluster_purity(true_labels, cluster_labels, cluster_label):
    # Get indices of data points belonging to the current cluster
    cluster_indices = np.where(cluster_labels == cluster_label)[0]
    # Get true labels of data points in the current cluster
    true_labels_in_cluster = true_labels[cluster_indices]
    # Count occurrences of each true label in the current cluster
    label_counts = Counter(true_labels_in_cluster)
    # Select the most frequent true label in the current cluster
    most_frequent_label = label_counts.most_common(1)[0][1]

    return most_frequent_label/len(cluster_indices), len(cluster_indices)

def calculate_purity(cluster_labels, true_labels):
    # Initialize total count of correctly classified data points
    total_correct = 0
    # Iterate through unique cluster labels
    for cluster_label in np.unique(cluster_labels):
        most_frequent_label, tmp = calculate_cluster_purity(true_labels, cluster_labels, cluster_label)
        # Increment total count of correctly classified data points
        total_correct += most_frequent_label*tmp

    return total_correct/len(cluster_labels)


def calculate_recall(eval_labels, clusters):
    cont_matrix = contingency_matrix(eval_labels, clusters)
    true_positives_per_cluster = cont_matrix.max(axis=0)
    true_labels_per_cluster = cont_matrix.sum(axis=0)
    false_negatives_per_cluster = true_labels_per_cluster - cont_matrix.max(axis=0)

    # Calculate recall for each cluster
    recall_per_cluster = true_positives_per_cluster / (true_positives_per_cluster + false_negatives_per_cluster)
    # Calculate weighted average recall
    weighted_avg_recall = (true_positives_per_cluster * recall_per_cluster).sum() / true_positives_per_cluster.sum()

    return weighted_avg_recall, recall_per_cluster


def calculate_F_measure(cluster_labels, true_labels):
    f_measure_sum = 0
    _, recalls = calculate_recall(true_labels, cluster_labels)
    for i,cluster_label in enumerate(np.unique(cluster_labels)):
        precision, _ = calculate_cluster_purity(true_labels, cluster_labels, cluster_label)
        recall = recalls[i]
        if precision + recall != 0:
            f_measure = (2 * precision * recall) / (precision + recall)
            f_measure_sum += f_measure
    f_measure_avg = f_measure_sum / 19
    return f_measure_avg


def cond_entropy(cluster_labels, true_labels):
    cond_entropy = 0
    for cluster in np.unique(cluster_labels):
        cluster_index = np.where(cluster == cluster_labels)[0]
        temp = np.zeros(19)
        for i in cluster_index:
            temp[true_labels[i] - 1] += 1
        temp = temp / len(cluster_index)

        temp2 = 0
        for i in range(19):
            if temp[i] != 0:
                temp2 -= temp[i] * math.log2(temp[i])
        cond_entropy += (temp2 * (len(cluster_index) / len(true_labels)))

    return cond_entropy


def Normalized_cut(data, k):
    similarity_matrix = kneighbors_graph(n_neighbors=19, X=data, mode='connectivity')
    similarity_matrix = sp.csc_matrix(similarity_matrix.toarray())
    dense_array = similarity_matrix.toarray()

    # Create a torch tensor from the dense NumPy array
    similarity_matrix = torch.tensor(dense_array)
    # similarity_matrix = torch.tensor(cosine_similarity(data))
    Delta = torch.diag(similarity_matrix.sum(dim=1))
    Delta_inverse = torch.inverse(Delta)
    laplacian_matrix = Delta_inverse @ (Delta - similarity_matrix)

    eigenvalues, eigenvectors = torch.linalg.eig(laplacian_matrix)
    # Extract real parts
    real_eigenvalues = eigenvalues.real
    real_eigenvectors = eigenvectors.real

    # sort indices
    sorted_indices = torch.argsort(real_eigenvalues)
    sorted_real_eigenvectors = real_eigenvectors[:, sorted_indices]

    # Get the first k sorted real eigenvectors
    first_k_eigenvectors = sorted_real_eigenvectors[:, :k]
    # Normalize the first k eigenvectors by row
    epsilon = 1e-5  # Small epsilon value to avoid division by very small numbers
    norms = torch.norm(first_k_eigenvectors, dim=1, keepdim=True)
    norms[norms < epsilon] = epsilon  # Replace very small norms with epsilon
    normalized_eigenvectors = first_k_eigenvectors / norms

    return normalized_eigenvectors


folder_path = "data"
train_data, train_labels, eval_data, eval_labels = load_data_split(folder_path)
scaler = StandardScaler()
k_means = KMeans(n_clusters=19)

# solution 1
solution1_train_data = np.mean(train_data, axis=1)
solution1_eval_data = np.mean(eval_data, axis=1)
solution1_train_data_scaled = scaler.fit_transform(solution1_train_data)
solution1_eval_data_scaled = scaler.fit_transform(solution1_eval_data)

# normalized_eigenvectors_train = Normalized_cut(solution1_train_data_scaled, 19)
normalized_eigenvectors_test = Normalized_cut(solution1_eval_data_scaled, 19)

# clusters_train1 = k_means.fit_predict(normalized_eigenvectors_train)
clusters_test1 = k_means.fit_predict(normalized_eigenvectors_test)
clusters_test1 += 1

# print("precision train: ", calculate_purity(new_clusters_train, train_labels))
print("precision test1: ", calculate_purity(clusters_test1, eval_labels))


# print("Weighted Average Recall train:", calculate_recall(train_labels, clusters_train1)[0])
print("Weighted Average Recall test1:", calculate_recall(eval_labels, clusters_test1)[0])

# print("f measure train1: ", calculate_F_measure(clusters_train1, train_labels))
print("f measure test1: ", calculate_F_measure(clusters_test1, eval_labels))

# print("cond entropy train1: ", cond_entropy(clusters_train1, train_labels))
print("cond entropy test1: ", cond_entropy(clusters_test1, eval_labels))
# # solution 2
# solution2_train_data = train_data.reshape(train_data.shape[0], -1)
# solution2_train_data_scaled = scaler.fit_transform(solution2_train_data)
# solution2_test_data = eval_data.reshape(eval_data.shape[0], -1)
# solution2_test_data_scaled = scaler.fit_transform(solution2_test_data)
#
# normlized_eigenvectors_train2 = Normalized_cut(solution2_train_data_scaled, 19)
# normlized_eigenvectors_test2 = Normalized_cut(solution2_test_data_scaled, 19)
#
# pca = PCA(n_components=100)
# solution2_data_pca = pca.fit_transform(solution2_train_data_scaled)
# solution2_test_data_pca = pca.fit_transform(solution2_test_data_scaled)
#
# clusters_train2 = k_means.fit_predict(normlized_eigenvectors_train2)
# clusters_test2 = k_means.fit_predict(normlized_eigenvectors_test2)
#
# new_clusters_train2 = map_clusters_to_labels(clusters_train2, train_labels)
# new_clusters_test2 = map_clusters_to_labels(clusters_test2, eval_labels)
#
# print("precision train2: ", calculate_purity(new_clusters_train2, train_labels))
# print("precision test2: ", calculate_purity(new_clusters_test2, eval_labels))
#
# print("Weighted Average Recall train:", calculate_recall(train_labels, clusters_train2))
# print("Weighted Average Recall test:", calculate_recall(eval_labels, clusters_test2))

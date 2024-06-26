import math
import os
import sys
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix
from sklearn.preprocessing import StandardScaler
from yellowbrick.cluster import SilhouetteVisualizer

k_values = [8, 13, 19, 28, 38]
EVAL_DATA_SIZE = 1824
TRAIN_DATA_SIZE = 7296
Sol1F_measure_train = []
Sol2F_measure_train = []
Sol1Entropy_train = []
Sol2Entropy_train = []
Sol1F_measure_test = []
Sol2F_measure_test = []
Sol1Entropy_test = []
Sol2Entropy_test = []
Sol1_Precisions_test = []
Sol1_Precisions_train = []
Sol1_Recalls_test = []
Sol1_Recalls_train = []
Sol2_Precisions_test = []
Sol2_Precisions_train = []
Sol2_Recalls_test = []
Sol2_Recalls_train = []
Initial_centroids = []
# Load data for training and evaluation
scaler = StandardScaler()


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
def Entropy(cluster, ground_truth):
    N = len(cluster)
    entropy = 0
    for partition in ground_truth:
        partition_set = set(map(tuple, partition))
        intersect = len(cluster.intersection(partition_set))
        if intersect != 0:
            entropy -= (intersect / N) * (math.log2(intersect / N))
    return entropy


def calculate_entropy(clusters, ground_truth, DATA_SIZE):
    Entropies = []
    entropy = 0
    for cluster in clusters:
        cluster_set = set(map(tuple, cluster))
        cluster_entropy = Entropy(cluster_set, ground_truth)
        Entropies.append(cluster_entropy)
        entropy += (len(cluster) / DATA_SIZE) * cluster_entropy
    return entropy, Entropies


def Precision(cluster, ground_truth):
    cluster_size = len(cluster)
    max_intersection = 0
    if cluster_size == 0:
        return 0
    for partition in ground_truth:
        partition_set = set(map(tuple, partition))
        intersect = len(cluster.intersection(partition_set))
        max_intersection = max(intersect, max_intersection)

    precision = max_intersection / cluster_size
    return precision


def Rec(cluster, ground_truth):
    max_intersection = 0
    size = 0
    if len(cluster) == 0:
        return 0
    for partition in ground_truth:
        partition_set = set(map(tuple, partition))
        intersect = len(cluster.intersection(partition_set))
        if intersect > max_intersection:
            size = len(partition)
            max_intersection = intersect
    recall = 0
    if size != 0:
        recall = max_intersection / size
    return recall


def calculate_purity(clusters, ground_truth, DATA_SIZE):
    purity = 0
    for cluster in clusters:
        cluster_set = set(map(tuple, cluster))
        purity += (len(cluster) / DATA_SIZE) * Precision(cluster_set, ground_truth)
    print("Purity =", purity)


def calculate_F_measure(clusters, ground_truth, DATA_SIZE):
    PRECISION = 0
    RECALL = 0
    Precisions = []
    Recalls = []
    F_measure = []
    clusters_sizes = []
    f = 0
    for cluster in clusters:
        clusters_sizes.append(len(cluster))
        cluster_set = set(map(tuple, cluster))
        precision = Precision(cluster_set, ground_truth)
        PRECISION = PRECISION + precision * (len(cluster) / DATA_SIZE)
        Precisions.append(precision)
        recall = Rec(cluster_set, ground_truth)
        RECALL = RECALL + recall * (len(cluster) / DATA_SIZE)
        Recalls.append(recall)
        f_measure = 0
        if precision + recall != 0:
            f_measure = (2 * precision * recall) / (precision + recall)
        F_measure.append(f_measure)
        f += f_measure
    f = f / len(clusters)

    return f, Precisions, Recalls, F_measure, PRECISION, RECALL, clusters_sizes


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


# Function to initialize centroids randomly
def initialize_centroids(data, k):
    centroids = np.empty((k, data.shape[1]))

    for i in range(k):
        # Select random data point as initial centroid
        centroid_idx = np.random.randint(0, len(data))
        centroids[i] = data[centroid_idx]

    Initial_centroids.append(centroids)
    return centroids


# K-Means clustering algorithm implementation
# K-Means clustering algorithm implementation
def kmeans(data, k):
    centroids = initialize_centroids(data, k)
    while True:  # Continue until convergence
        # Assign data points to closest centroids
        clusters_points = [[] for _ in range(k)]
        clusters = []
        for point in data:
            distances = np.array([euclidean_distance(point, centroid) for centroid in centroids])
            cluster_idx = np.argmin(distances)
            clusters.append(cluster_idx)
            clusters_points[cluster_idx].append(point)
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
                new_centroids[i] = None  # not Possible
        # Check convergence by comparing centroids
        if np.all(centroids == new_centroids):
            break  # Break if centroids have not changed
        # Update centroids
        centroids = new_centroids.copy()
    return centroids, clusters_points, clusters


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


def Evaluate_Sol1(clusters_train, ground_truth_train, clusters_test, ground_truth_test,k,train_labels, eval_labels,labels_train,labels_test):
    f, Precisions, Recalls, F_measures, p, r, clusters_sizes = calculate_F_measure(clusters_test, ground_truth_test,
                                                                                   EVAL_DATA_SIZE)
    entropy, Entropies = calculate_entropy(clusters_test, ground_truth_test, EVAL_DATA_SIZE)
    Sol1F_measure_test.append(f)
    Sol1Entropy_test.append(entropy)
    Sol1_Precisions_test.append(p)
    Sol1_Recalls_test.append(r)
    print("Eval Data Results for k : ", k)
    print("Precision = ", p)
    print("Recall = ", r)
    recall,_=calculate_recall(eval_labels, labels_test)
    print("Recall 2 = ", recall)
    print("F-measure = ", f)
    print("Entropy =", entropy)
    # tabulate(f, Precisions, Recalls, F_measures, entropy, Entropies, clusters_sizes)
    print(
        "---------------------------------------------------------------------")
    f, Precisions, Recalls, F_measures, p, r, clusters_sizes = calculate_F_measure(clusters_train, ground_truth_train,
                                                                                   TRAIN_DATA_SIZE)
    entropy, Entropies = calculate_entropy(clusters_train, ground_truth_train, TRAIN_DATA_SIZE)
    Sol1F_measure_train.append(f)
    Sol1Entropy_train.append(entropy)
    Sol1_Precisions_train.append(p)
    Sol1_Recalls_train.append(r)
    print("Train Data Results for k : ", k)
    print("Precision = ", p)
    print("Recall = ", r)
    recall, _ = calculate_recall(train_labels, labels_train)
    print("Recall 2 = ", recall)
    print("F-measure = ", f)
    print("Entropy =", entropy)
    # tabulate(f, Precisions, Recalls, F_measures, entropy, Entropies, clusters_sizes)


def Evaluate_Sol2(clusters_train, ground_truth_train, clusters_test, ground_truth_test,k,train_labels, eval_labels,labels_train,labels_test):
    f, Precisions, Recalls, F_measures, p, r, clusters_sizes = calculate_F_measure(clusters_test, ground_truth_test,
                                                                                   EVAL_DATA_SIZE)
    entropy, Entropies = calculate_entropy(clusters_test, ground_truth_test, EVAL_DATA_SIZE)
    Sol2F_measure_test.append(f)
    Sol2Entropy_test.append(entropy)
    Sol2_Precisions_test.append(p)
    Sol2_Recalls_test.append(r)
    print("Eval Data Results for k :", k)
    print("Precision = ", p)
    print("Recall = ", r)
    recall, _ = calculate_recall(eval_labels, labels_test)
    print("Recall 2 = ", recall)
    print("F-measure = ", f)
    print("Entropy =", entropy)
    # tabulate(f, Precisions, Recalls, F_measures, entropy, Entropies, clusters_sizes)
    print(
        "---------------------------------------------------------------------")
    f, Precisions, Recalls, F_measures, p, r, clusters_sizes = calculate_F_measure(clusters_train, ground_truth_train,
                                                                                   TRAIN_DATA_SIZE)
    entropy, Entropies = calculate_entropy(clusters_train, ground_truth_train, TRAIN_DATA_SIZE)
    Sol2F_measure_train.append(f)
    Sol2Entropy_train.append(entropy)
    Sol2_Precisions_train.append(p)
    Sol2_Recalls_train.append(r)
    print("Train Data Results for k : ", k)
    print("Precision = ", p)
    print("Recall = ", r)
    recall, _ = calculate_recall(train_labels, labels_train)
    print("Recall 2 = ", recall)
    print("F-measure = ", f)
    print("Entropy =", entropy)
    # tabulate(f, Precisions, Recalls, F_measures, entropy, Entropies, clusters_sizes)


def prepareData_Sol1(train_data, eval_data):
    solution1_train_data = np.mean(train_data, axis=1)
    solution1_eval_data = np.mean(eval_data, axis=1)
    solution1_train_data_scaled = scaler.fit_transform(solution1_train_data)
    solution1_eval_data_scaled = scaler.fit_transform(solution1_eval_data)
    solution1_ground_truth_test = extract_ground_truth(solution1_eval_data_scaled)
    solution1_ground_truth_train = extract_ground_truth(solution1_train_data_scaled, 384)
    return solution1_train_data_scaled, solution1_eval_data_scaled, solution1_ground_truth_test, solution1_ground_truth_train


def Sol1_for_k(solution1_train_data_scaled, solution1_eval_data_scaled, solution1_ground_truth_test,
               solution1_ground_truth_train, k,train_labels, eval_labels):
    centroids, clusters_train,labels_train = kmeans(solution1_train_data_scaled, k)
    labels_test, clusters_test = assign_to_nearest_centroid(solution1_eval_data_scaled, centroids)
    # calculate_purity(clusters, solution1_ground_truth)
    print("Evaluation Results for k : ", k)
    Evaluate_Sol1(clusters_train, solution1_ground_truth_train, clusters_test, solution1_ground_truth_test, k,train_labels, eval_labels,labels_train,labels_test)
    return centroids


def prepareData_Sol2(train_data, eval_data):
    solution2_train_data = train_data.reshape(train_data.shape[0], -1)
    solution2_eval_data = eval_data.reshape(eval_data.shape[0], -1)
    solution2_train_data_scaled = scaler.fit_transform(solution2_train_data)
    solution2_eval_data_scaled = scaler.fit_transform(solution2_eval_data)
    pca = PCA(n_components=0.85)
    solution2_train_data_pca = pca.fit_transform(solution2_train_data_scaled)
    solution2_eval_data_pca = pca.transform(solution2_eval_data_scaled)
    solution2_ground_truth_test = extract_ground_truth(solution2_eval_data_pca)
    solution2_ground_truth_train = extract_ground_truth(solution2_train_data_pca, 384)
    return solution2_train_data_pca, solution2_eval_data_pca, solution2_ground_truth_test, solution2_ground_truth_train


def Sol2_for_k(solution2_train_data_pca, solution2_eval_data_pca, solution2_ground_truth_test,
               solution2_ground_truth_train, k,train_labels, eval_labels):
    centroids, clusters_train ,labels_train= kmeans(solution2_train_data_pca, k)
    labels_test, clusters_test = assign_to_nearest_centroid(solution2_eval_data_pca, centroids)
    # calculate_purity(clusters, solution2_ground_truth)
    print("Evaluation Results for k : ", k)
    Evaluate_Sol2(clusters_train, solution2_ground_truth_train, clusters_test, solution2_ground_truth_test, k,train_labels, eval_labels,labels_train,labels_test)
    return centroids


def tabulate(total_f_measure, Precisions, Recalls, F_measure, entropy, Entropies, cluster_sizes):
    cluster_names = [f"Cluster {i + 1}" for i in range(len(Precisions))]
    total_size = sum(cluster_sizes)
    sizes_ratio = [str("{:.4f}".format((size / total_size)*100))+" %" for size in cluster_sizes]
    df = pd.DataFrame({'Cluster Size': cluster_sizes, 'Size Ratio': sizes_ratio}, index=cluster_names)
    df.loc['Total Size '] = [total_size, '']
    print(df)
    df = pd.DataFrame({'Precision': Precisions, 'Recall': Recalls, 'F-Measure': F_measure}, index=cluster_names)
    df.loc['Total'] = ['', '', total_f_measure]
    print(df)
    df = pd.DataFrame({'Conditional Entropy': Entropies}, index=cluster_names)
    df.loc['Total'] = [entropy]
    print(df)

# 93 %
def Silhouette(data, centroids, k):
    kmeans_model = KMeans(n_clusters=k, init=centroids, n_init=1)
    visualizer = SilhouetteVisualizer(kmeans_model, colors='yellowbrick')
    visualizer.fit(data)
    visualizer.show()


def plot_results(k_values, Sol1F_measure, Sol2F_measure, Sol1Entropy, Sol2Entropy, label, y1, y2):
    import numpy as np
    import matplotlib.pyplot as plt

    # Calculate bar width
    bar_width = 0.35

    # Set positions for the bars
    bar_positions_1 = np.arange(len(k_values))
    bar_positions_2 = [x + bar_width for x in bar_positions_1]

    # Define colors for bars
    color1 = 'blue'
    color2 = 'orange'

    # Create subplots
    fig, ax = plt.subplots(2, 1, figsize=(10, 10))

    # Plot for y1
    ax[0].bar(bar_positions_1, Sol1F_measure, bar_width, label='Solution 1', color=color1)
    ax[0].bar(bar_positions_2, Sol2F_measure, bar_width, label='Solution 2', color=color2)
    ax[0].set_title(y1 + ' vs. k ' + label)
    ax[0].set_xlabel('k')
    ax[0].set_ylabel(y1)
    ax[0].set_xticks(bar_positions_1 + bar_width / 2)
    ax[0].set_xticklabels(k_values)
    ax[0].legend()

    # Plot for y2
    ax[1].bar(bar_positions_1, Sol1Entropy, bar_width, label='Solution 1', color=color1)
    ax[1].bar(bar_positions_2, Sol2Entropy, bar_width, label='Solution 2', color=color2)
    ax[1].set_title(y2 + ' vs. k ' + label)
    ax[1].set_xlabel('k')
    ax[1].set_ylabel(y2)
    ax[1].set_xticks(bar_positions_1 + bar_width / 2)
    ax[1].set_xticklabels(k_values)
    ax[1].legend()

    plt.tight_layout()
    plt.show()



def calculate_Recall(cluster_labels, ground_truth_labels):
    N = len(cluster_labels)
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(N):
        for j in range(i + 1, N):
            if cluster_labels[i] == cluster_labels[j] and ground_truth_labels[i] == ground_truth_labels[j]:
                TP = TP + 1
            elif cluster_labels[i] == cluster_labels[j] and ground_truth_labels[i] != ground_truth_labels[j]:
                FP = FP + 1
            elif cluster_labels[i] != cluster_labels[j] and ground_truth_labels[i] == ground_truth_labels[j]:
                FN = FN + 1
            elif cluster_labels[i] != cluster_labels[j] and ground_truth_labels[i] != ground_truth_labels[j]:
                TN = TN + 1
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)



train_data, train_labels, eval_data, eval_labels = load_data_split(folder_path)
solution1_train_data_scaled, solution1_eval_data_scaled, solution1_ground_truth_test, solution1_ground_truth_train = prepareData_Sol1(
    train_data,
    eval_data)
solution2_train_data_pca, solution2_eval_data_pca, solution2_ground_truth_test, solution2_ground_truth_train = prepareData_Sol2(
    train_data, eval_data)
print("Solution 1:")
for k in k_values:
    print(
        "-----------------------------------------------------------------------------------------------------------------")
    centroids = Sol1_for_k(solution1_train_data_scaled, solution1_eval_data_scaled, solution1_ground_truth_test,
                           solution1_ground_truth_train, k,train_labels, eval_labels)
    Silhouette(solution1_train_data_scaled, centroids, k)

print("-----------------------------------------------------------------------------------------------------------------")
print("-----------------------------------------------------------------------------------------------------------------")
print("Solution 2:")
for k in k_values:
    print(
        "-----------------------------------------------------------------------------------------------------------------")
    centroids = Sol2_for_k(solution2_train_data_pca, solution2_eval_data_pca, solution2_ground_truth_test,
                           solution2_ground_truth_train, k,train_labels, eval_labels)
    Silhouette(solution2_train_data_pca, centroids, k)
plot_results(k_values, Sol1_Precisions_test, Sol2_Precisions_test, Sol1_Recalls_test, Sol2_Recalls_test, "Evaluation "
                                                                                                         "Data",
             "Precision", "Recall")
plot_results(k_values, Sol1_Precisions_train, Sol2_Precisions_train, Sol1_Recalls_train, Sol2_Recalls_train,
             "Training Data", "Precision", "Recall")
plot_results(k_values, Sol1F_measure_test, Sol2F_measure_test, Sol1Entropy_test, Sol2Entropy_test, "Evaluation Data",
             "F-measure", "Entropy")
plot_results(k_values, Sol1F_measure_train, Sol2F_measure_train, Sol1Entropy_train, Sol2Entropy_train, "Training Data",
             "F-measure", "Entropy")

##

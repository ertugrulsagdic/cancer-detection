import numpy as np

class K_Means:
    def __init__(self, k=3, max_iters=500):
        self.K = k
        self.max_iters = max_iters

        # list of sample indices for each cluster
        # For each cluster, init. an empty list.
        self.clusters = [[] for _ in range(self.K)]
        self.centroids = []  # mean vectors are stored

    def predict(self,data):
        n_samples, n_features = data.shape

        # Initialize centroids
        random_sample_indexes = np.random.choice(n_samples, self.K, replace=False)
        self.centroids = [data[idx] for idx in random_sample_indexes]

        # Optimization
        for my_iter in range(self.max_iters):
            # update clusters
            self.clusters = self.create_clusters(self.centroids, data)
            # update centroids
            old_centroids = self.centroids
            self.centroids = self.get_centroids(self.clusters, n_features, data)  # Assign mean value for each cluster
            # check if converged
            if self.is_converged(old_centroids, self.centroids):
                break

        # return cluster labels
        return self.get_cluster_labels(self.clusters, n_samples)

    @staticmethod
    def get_cluster_labels(clusters, n_samples):
        labels = np.empty(n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    def create_clusters(self, centroids, data):
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(data):
            centroid_idx = self.closest_centroid(self, sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    @staticmethod
    def closest_centroid(self, sample, centroids):
        distances = [self.euc_dist(sample, point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    def get_centroids(self, clusters, n_features, data):
        centroids = np.zeros((self.K, n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(data[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_converged(self, old_centroids, new_centroids):
        distances = [self.euc_dist(old_centroids[i], new_centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def euc_dist(self, A, B):
        return np.sqrt(np.sum((A-B)**2))
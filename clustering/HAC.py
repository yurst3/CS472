import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from math import sqrt
from tqdm import tqdm

class HACClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,link_type='single'): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            link_type = single or complete. when combining two clusters use complete link or single link
        """
        self.link_type = link_type
        self.k = k
    def fit(self, X, b=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            b (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        self.data = X

        # Create adjacency (distance) matrix using Euclidean distance
        # Sqrt(sum([(item_a[col] - item_b[col])**2]))
        distance_matrix = [[sqrt(sum([(item_a[col] - item_b[col])**2 for col in range(X.shape[1])]))
                   if i_a > i_b else float('inf') for i_b, item_b in enumerate(X)] for i_a, item_a in enumerate(X)]
        distance_matrix = np.array(distance_matrix)

        # Map of dictionary indices --> cluster
        distance_cluster = {i: tuple([i]) for i in range(X.shape[0])}
        # Map of cluster --> instances
        self.cluster_instances = {tuple([index]): set(map(tuple, np.expand_dims(value, axis=0))) for index, value in enumerate(X)}

        # While we have more than the desired number of clusters
        with tqdm(total=X.shape[0] - self.k) as pbar:
            while len(self.cluster_instances) > self.k:

                # Get row/column indices of the min distance
                b, a = np.unravel_index(distance_matrix.argmin(), distance_matrix.shape)

                # Perform key lookup with for item a and item b
                b_key = distance_cluster[b]
                a_key = distance_cluster[a]

                # Perform set lookup with each key
                b_set = self.cluster_instances[b_key]
                a_set = self.cluster_instances[a_key]

                # Join the keys and the sets
                key_join = b_key + a_key
                set_join = b_set.union(a_set)

                # Remove old (cluster --> instances) mapping and add new one
                del self.cluster_instances[b_key]
                del self.cluster_instances[a_key]
                self.cluster_instances[key_join] = set_join

                # Remove old (distance matrix index --> cluster) mapping and add new one
                del distance_cluster[b]
                distance_cluster[a] = key_join

                # Update distances from cluster (a,b) to all other nodes
                for i in range(X.shape[0]):
                    dis_b = distance_matrix[b][i] if i < b else distance_matrix[i][b]
                    if i < a:
                        dis_a = distance_matrix[a][i]
                        distance_matrix[a][i] = min((dis_a, dis_b)) if self.link_type is 'single' else max((dis_a, dis_b))
                    elif i > a:
                        dis_a = distance_matrix[i][a]
                        distance_matrix[i][a] = min((dis_a, dis_b)) if self.link_type is 'single' else max((dis_a, dis_b))

                # Set row and column corresponding to item b to infinity
                distance_matrix[b,:] = float('inf')
                distance_matrix[:,b] = float('inf')

                pbar.update(1)

        return self

    def _calc_centroid(self, cluster):
        arr = np.array(list(cluster))

        return np.array([sum(arr[:, i])/arr.shape[0] for i in range(arr.shape[1])])

    def _calc_SSE(self, centroid, cluster):
        SSE = 0
        cluster_arr = np.array(list(cluster))

        for instance in cluster_arr:
            diffs = [(centroid[i] - instance[i])**2 for i in range(cluster_arr.shape[1])]
            SSE += sum(diffs)

        return SSE

    def save_clusters(self,filename):
        """
            f = open(filename,"w+") 
            Used for grading.
            write("{:d}\n".format(k))
            write("{:.4f}\n\n".format(total SSE))
            for each cluster and centroid:
                write(np.array2string(centroid,precision=4,separator=","))
                write("\n")
                write("{:d}\n".format(size of cluster))
                write("{:.4f}\n\n".format(SSE of cluster))
            f.close()
        """
        SSEs = []
        cluster_lens = []
        centroids = []

        # Calculate the centroid, SSE, and length of each cluster
        # Append to the appropriate list
        for cluster in self.cluster_instances.values():
            centroids.append(self._calc_centroid(cluster))
            SSEs.append(self._calc_SSE(centroids[-1], cluster))
            cluster_lens.append(len(cluster))

        # Print in the weird formatting requirments
        f = open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(sum(SSEs)))

        for centroid, SSE, cluster_len in zip(centroids, SSEs, cluster_lens):
            f.write(np.array2string(centroid, precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(cluster_len))
            f.write("{:.4f}\n\n".format(SSE))

        f.close()

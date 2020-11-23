import numpy as np
from math import sqrt
from sklearn.base import BaseEstimator, ClusterMixin
from random import uniform

class KMEANSClustering(BaseEstimator,ClusterMixin):

    def __init__(self,k=3,debug=False): ## add parameters here
        """
        Args:
            k = how many final clusters to have
            debug = if debug is true use the first k instances as the initial centroids otherwise choose random points as the initial centroids.
        """
        self.k = k
        self.debug = debug

    def fit(self,X,y=None):
        """ Fit the data; In this lab this will make the K clusters :D
        Args:
            X (array-like): A 2D numpy array with the training data
            y (array-like): An optional argument. Clustering is usually unsupervised so you don't need labels
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """
        if self.debug is True:
            # Use the first k instances as centroids
            self.centroids = [tuple(X[k]) for k in range(self.k)]
        else:
            # Pick k random centroids with attributes between the min & max of all instance attributes
            self.centroids = [tuple([uniform(min(X[:,i]), max(X[:,i])) for i in range(X.shape[0])]) for k in range(self.k)]

        # Clusters from previous iterations
        previous = None
        self.clusters = None

        # Stop if the previous clusters and current clusters are the same
        while self._stop(previous, self.clusters):
            # Previous is the same as all of the clusters before being updated
            previous = self.clusters.copy() if self.clusters is not None else dict(zip(self.centroids, [set([]) for i in range(self.k)]))
            self.clusters = dict(zip(self.centroids, [set([]) for i in range(self.k)]))

            # Group all instances with their nearest centroid
            for instance in X:
                # Create map of euclidean distances to each centroid
                distances = {self._euclid_distance(instance, centroid): centroid for centroid in self.centroids}

                # Assign instance to closest centroid
                closest_centroid = distances[min(distances.keys())]
                self.clusters[closest_centroid].add(tuple(instance))

            # Recalculate each centroid to be at the center of its cluster
            for i in range(len(self.centroids)):
                cluster = self.clusters[self.centroids[i]]
                cluster_arr = np.array(list(cluster))

                # Only recalculate if there are points grouped with the centroid
                if len(cluster) > 0:
                    self.centroids[i] = tuple([self._calc_attr(cluster_arr[:, attribute]) for attribute in range(X.shape[1])])

        return self

    def _calc_attr(self, instances):
        return sum(instances)/len(instances)

    def _euclid_distance(self, x, y):
        return sqrt(sum([(x[i] - y[i])**2 for i in range(x.shape[0])]))

    def _stop(self, prev, cur):
        # Prev is None on the first iteration
        if prev is None or cur is None:
            return True
        else:
            # Loop through all of the clusters
            for prev_vals, cur_vals in zip(prev.values(),cur.values()):
                # Check if previous cluster and current cluster are NOT equal to each other
                if not (prev_vals.issubset(cur_vals) and cur_vals.issubset(prev_vals)):
                    return True
            # If all are equal to each other, return False
            return False

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
        # Calculate the centroid, SSE, and length of each cluster
        # Append to the appropriate list

        SSEs = []
        for centroid in self.clusters:
            SSEs.append(self._calc_SSE(centroid, self.clusters[centroid]))

        # Print in the weird formatting requirements
        f = open(filename, "w+")
        f.write("{:d}\n".format(self.k))
        f.write("{:.4f}\n\n".format(sum(SSEs)))

        for centroid, SSE in zip(self.centroids, SSEs):
            f.write(np.array2string(np.array(centroid), precision=4, separator=","))
            f.write("\n")
            f.write("{:d}\n".format(len(self.clusters[centroid])))
            f.write("{:.4f}\n\n".format(SSE))

        f.close()
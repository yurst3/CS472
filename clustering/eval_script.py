from clustering.HAC import HACClustering
from clustering.Kmeans import KMEANSClustering
from tools.arff import Arff
import numpy as np

abalone = "../data/clustering/abalone.arff"

mat = Arff(abalone, label_count=0) ## label_count = 0 because clustering is unsupervised.

raw_data = mat.data
norm_data = raw_data

# min/max normalize the data
for col in range(norm_data.shape[1]):
    column = norm_data[:, col]
    col_min = min(column)
    col_max = max(column)

    where = np.where(True, (column - col_min) / (col_max - col_min), 0)

    norm_data[:, col] = where

### KMEANS ###
KMEANS = KMEANSClustering(k=5, debug=True)
KMEANS.fit(norm_data)
KMEANS.save_clusters("debug_kmeans.txt")

### HAC SINGLE LINK ###
HAC_single = HACClustering(k=5, link_type='single')
HAC_single.fit(norm_data)
HAC_single.save_clusters("debug_hac_single.txt")

### HAC COMPLETE LINK ###
HAC_complete = HACClustering(k=5,link_type='complete')
HAC_complete.fit(norm_data)
HAC_complete.save_clusters("debug_hac_complete.txt")
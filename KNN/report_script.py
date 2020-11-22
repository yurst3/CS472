from KNN import KNNClassifier
from tools.arff import Arff
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

seismic_train = "../data/KNN/seismic-bumps_train.arff"
seismic_test = "../data/KNN/seismic-bumps_test.arff"
magic_telescope_train = "../data/KNN/magic-telescope_train.arff"
magic_telescope_test = "../data/KNN/magic-telescope_test.arff"
housing_train = "../data/KNN/housing-price_train.arff"
housing_test = "../data/KNN/housing-price_test.arff"
credit_approval = "../data/KNN/credit-approval.arff"

mat = Arff(credit_approval, label_count=1)
#mat2 = Arff(housing_test, label_count=1)
raw_data = mat.data

split = 0.75

train_data = raw_data[:int(raw_data.shape[0]*split), :-1]
norm_train_data = train_data.copy()
train_labels = raw_data[:int(raw_data.shape[0]*split), -1]

for col in range(norm_train_data.shape[1]):
    column = norm_train_data[:, col]
    col_min = min(column)
    col_max = max(column)

    where = np.where(True, (column - col_min) / (col_max - col_min), 0)

    norm_train_data[:, col] = where

test_data = raw_data[int(raw_data.shape[0]*split)+1:, :-1]
norm_test_data = test_data.copy()
test_labels = raw_data[int(raw_data.shape[0]*split)+1:, -1]

for col in range(norm_test_data.shape[1]):
    column = norm_test_data[:, col]
    col_min = min(column)
    col_max = max(column)

    where = np.where(True, (column - col_min) / (col_max - col_min), 0)

    norm_test_data[:, col] = where

KNN_weight = KNNClassifier(k_val=15,
                           label_type='nominal',
                           col_types=mat.attr_types,
                           weight_type='inverse_distance')
KNN_weight.fit(norm_train_data, train_labels)
weight_scores = KNN_weight.score(norm_test_data, test_labels)

K_vals = np.arange(1,17,2)

#plt.plot(K_vals, scores, label="non-weighted")
plt.plot(K_vals, weight_scores, label="weighted")
plt.title("Credit Approval")
plt.ylabel("Accuracy")
plt.xlabel("K nearest neighbors")
plt.savefig("part5_plot_credit.png")
plt.show()
from decisiontree import DTClassifier
from tools.arff import Arff
import random
from tqdm import tqdm
import numpy as np
from sklearn import tree

cars = "../data/decisiontree/cars.arff"
voting = "../data/decisiontree/voting.arff"

mat = Arff(cars, missing=2.0)

counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]

np_mat = mat.data
#random.shuffle(np_mat)

total = np_mat.shape[0]

accuracies = []

# K-fold cross validation
K = 10
with tqdm(total=K) as pbar:
       for i in range(K):
              block = int(total / K)

              # [<----- training data -----> | <- testing data -> | <--------- training data ---------> ]
              training_data = np_mat[:i * block, 0:-1]
              training_labels = np_mat[:i * block, -1].reshape(-1, 1)

              if i < K - 1:
                     data_remainder = np_mat[((i + 1) * block) + 1:, 0:-1]
                     label_remainder = np_mat[((i + 1) * block) + 1:, -1].reshape(-1, 1)

                     training_data = np.concatenate((training_data, data_remainder), axis=0)
                     training_labels = np.concatenate((training_labels, label_remainder), axis=0)

              test_data = np_mat[(i * block) + 1:(i + 1) * block, 0:-1]
              test_labels = np_mat[(i * block) + 1:(i + 1) * block, -1].reshape(-1, 1)

              #DTClass = DTClassifier(counts)
              DTClass = tree.DecisionTreeClassifier()
              DTClass.fit(training_data,training_labels)

              accuracies.append(DTClass.score(test_data, test_labels))

              pbar.set_description(f"Accuracy: {accuracies[-1]:.2f}")

              pbar.update(1)

tree.export_graphviz(DTClass, out_file="tree.dot", max_depth=5)

print("Average Accuracy = [{:.2f}]\n".format(sum(accuracies)/len(accuracies)))
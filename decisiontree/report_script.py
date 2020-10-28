from decisiontree import DTClassifier
from tools.arff import Arff
import random
import numpy as np

cars = "../data/decisiontree/cars.arff"
vowel = "../data/decisiontree/vowel.arff"
mat = Arff(cars)

counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]

np_mat = mat.data
random.shuffle(np_mat)

total = np_mat.shape[0]
split = int(total * 0.75)

training_data = np_mat[:split,0:-1]
training_labels = np_mat[:split,-1].reshape(-1,1)
test_data = np_mat[split+1:,0:-1]
test_labels = np_mat[split+1:,-1].reshape(-1,1)

DTClass = DTClassifier(counts)
DTClass.fit(training_data,training_labels)

Acc = DTClass.score(test_data, test_labels)

print("Accuracy = [{:.2f}]".format(Acc))
from tools.arff import Arff
from backpropagation.mlp import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import random

debug = "../data/perceptron/debug/linsep2nonorigin.arff"
eval = "../data/perceptron/evaluation/data_banknote_authentication.arff"

mat = Arff(debug, label_count=1)
np_mat = mat.data

total = np_mat.shape[0]
split = int(total * 0.7)

training_data = mat[0:split, :-1]
test_data = mat[split+1:, :-1]
training_labels = mat[0:split,-1].reshape(-1,1)
test_labels = mat[split+1:,-1].reshape(-1,1)

# [3,3] = 2 hidden layers with 3 nodes
mlp = MLPClassifier(lr=.2, momentum=.5, shuffle=False, hidden_layer_widths=[3, 3])
mlp.fit(training_data, training_labels)

from tools.arff import Arff
from backpropagation.mlp import MLPClassifier
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

debug = "../data/perceptron/debug/linsep2nonorigin.arff"
eval = "../data/perceptron/evaluation/data_banknote_authentication.arff"

mat = Arff(eval, label_count=1)
np_mat = mat.data

total = np_mat.shape[0]
split = int(total * 1)

training_data = mat[0:split, :-1]
test_data = mat[split+1:, :-1]
training_labels = mat[0:split,-1].reshape(-1,1)
test_labels = mat[split+1:,-1].reshape(-1,1)

mlp = MLPClassifier(lr=.1, momentum=.5, shuffle=False, hidden_layer_widths=[training_data.shape[1] * 2])

mlp.fit(training_data, training_labels)
binary_weights = mlp.get_weights()

my_dict = dict(Binary=binary_weights)

df = pd.DataFrame(dict([ (k,pd.Series(v)) for k,v in my_dict.items() ]))
print(df)
df.to_csv("evaluation.csv")
from tools.arff import Arff
from backpropagation.mlp import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd

iris = "../data/backpropagation/iris.arff"
vowel = "../data/backpropagation/vowel.arff"

mat = Arff(vowel, label_count=1)
np_mat = mat.data

random.seed(125)
random.shuffle(np_mat)

total = np_mat.shape[0]
split = int(total * 0.75)

training_data = mat[0:split, 1:-1]
test_data = mat[split+1:, 1:-1]
training_labels = LabelBinarizer().fit_transform(pd.DataFrame(mat[0:split,-1].reshape(-1,1)))
test_labels = LabelBinarizer().fit_transform(pd.DataFrame(mat[split+1:,-1].reshape(-1,1)))

results = []

learning_rates = np.linspace(0.1, 1, 10)
repeat = 10

for lr in learning_rates:
    avg = []
    for i in range(repeat):
        mlp = MLPClassifier(lr=lr, momentum=.9, shuffle=True, hidden_layer_widths=[training_data.shape[1] * 2])
        trains, vals = mlp.fit(training_data, training_labels)
        MSE, accuracy = mlp.score(test_data, test_labels)

        avg.append([trains[-1], vals[-1][0], MSE])

    avg = np.array(avg)
    results.append([sum(avg[:,0])/repeat, sum(avg[:,1])/repeat, sum(avg[:,2])/repeat])

    print("Lr: ", lr, " Avg MSE: ", results[-1][2])

results = np.array(results)

plt.plot(learning_rates, results[:,0], label="Train Set MSE")
plt.plot(learning_rates, results[:,1], label="Val Set MSE")
plt.plot(learning_rates, results[:,2], label="Test Set MSE")
plt.legend()
plt.title("Final MSE At Different Learning Rates")
plt.ylabel("MSE")
plt.xlabel("Learning Rate")
plt.show()
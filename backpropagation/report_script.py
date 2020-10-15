from tools.arff import Arff
from backpropagation.mlp import MLPClassifier
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import numpy as np
import random
import pandas as pd
from tqdm import tqdm

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

momentums = np.linspace(0, 1, 11)
repeat = 5

with tqdm(total=(len(momentums) * repeat)) as pbar:
    for momentum in momentums:
        avg = []
        for i in range(repeat):
            mlp = MLPClassifier(lr=0.1, momentum=momentum, shuffle=True, hidden_layer_widths=[64])
            trains, vals = mlp.fit(training_data, training_labels)
            MSE, accuracy = mlp.score(test_data, test_labels)

            avg.append([trains[-1], vals[-1][0], MSE, accuracy, len(trains)])
            pbar.update(1)

        avg = np.array(avg)
        results.append([sum(avg[:,0])/repeat,
                        sum(avg[:,1])/repeat,
                        sum(avg[:,2])/repeat,
                        sum(avg[:,3])/repeat,
                        sum(avg[:,4])/repeat])

        pbar.set_description(f"Nodes: {momentum}, Avg Accuracy: {results[-1][3]}")

results = np.array(results)
np.save("results.npy", results)

#results = np.load("results.npy")

plt.plot(momentums, results[:, 0], label="Train Set MSE")
plt.plot(momentums, results[:, 1], label="Val Set MSE")
plt.plot(momentums, results[:, 2], label="Test Set MSE")
plt.legend()
plt.title("Final MSE At Different Momentums")
plt.ylabel("Final MSE")
plt.xlabel("Momentum")
plt.show()
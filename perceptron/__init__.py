from CS472.tools.arff import Arff
from CS472.perceptron.perceptron import PerceptronClassifier
import matplotlib.pyplot as plt
import numpy as np
import random

eval = "data/perceptron/evaluation/data_banknote_authentication.arff"
mat = Arff(eval,label_count=1)
np_mat = mat.data

total = np_mat.shape[0]
split = int(total * 0.7)

random.shuffle(np_mat)

# Separate training from test data 70/30 split
training_data = mat[0:split, :-1]
test_data = mat[split+1:, :-1]
training_labels = mat[0:split,-1].reshape(-1,1)
test_labels = mat[split+1:,-1].reshape(-1,1)

P2Class = PerceptronClassifier(lr=0.1,shuffle=False)
epochs, miss_rate, pc = P2Class.fit(training_data,training_labels)
test_accuracy = P2Class.score(test_data, test_labels)
training_accuracy = P2Class.score(training_data, training_labels)

weights = P2Class.get_weights()
print("Test Accuracy = [{:.2f}]".format(test_accuracy))
print("Training Accuracy = [{:.2f}]".format(training_accuracy))
print("Final Weights =",weights)
print("Epochs =",epochs)

# Plot test data points
for i in range(test_labels.shape[0]):
    if test_labels[i][0] == 0:
        plt.scatter(test_data[i,0], test_data[i,1], color=(1,0,0), label=0)
    else:
        plt.scatter(test_data[i,0], test_data[i,1], color=(0,0,1), label=1)

# Calculate/plot line
slope = -1 * weights[0] / weights[1]
intercept = -1 * weights[2] / weights[1]
axes = plt.gca()
x_vals = np.array(axes.get_xlim())
y_vals = intercept + slope * x_vals
plt.plot(x_vals, y_vals, '--')
#plt.plot(miss_rate)

plt.title('Average Misclassification Rate Across Epochs')
plt.xlabel('Epoch')
plt.ylabel('Misclassification Rate')
plt.show()
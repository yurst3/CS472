from KNN import KNNClassifier
from tools.arff import Arff
import numpy as np

diabetes_train = "../data/KNN/diabetes.arff"
diabetes_test = "../data/KNN/diabetes_test.arff"
seismic_train = "../data/KNN/seismic-bumps_train.arff"
seismic_test = "../data/KNN/seismic-bumps_test.arff"

mat = Arff(seismic_train, label_count=1)
mat2 = Arff(seismic_test, label_count=1)
raw_data = mat.data
h, w = raw_data.shape
train_data = raw_data[:, :-1]
train_labels = raw_data[:, -1]

raw_data2 = mat2.data
h2, w2 = raw_data2.shape
test_data = raw_data2[:, :-1]
test_labels = raw_data2[:, -1]

KNN = KNNClassifier(15, "nominal", weight_type='inverse_distance')
KNN.fit(train_data, train_labels)
pred = KNN.predict(test_data)
score = KNN.score(test_data, test_labels)
print(f"Score: {score[1]*100:.2f}%")
#np.savetxt("diabetes-prediction.csv", pred, delimiter=',', fmt="%i")
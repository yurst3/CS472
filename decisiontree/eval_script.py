from decisiontree import DTClassifier
from tools.arff import Arff
import numpy as np

mat = Arff("../data/decisiontree/lenses.arff")

counts = [] ## this is so you know how many types for each column

for i in range(mat.data.shape[1]):
       counts += [mat.unique_value_count(i)]
data = mat.data[:,0:-1]
labels = mat.data[:,-1].reshape(-1,1)

DTClass = DTClassifier(counts)
DTClass.fit(data,labels)

mat2 = Arff("../data/decisiontree/all_lenses.arff")
data2 = mat2.data[:,0:-1]
labels2 = mat2.data[:,-1]
pred = DTClass.predict(data2)
Acc = DTClass.score(data2,labels2)

np.savetxt("pred_lenses.csv",pred,delimiter=",")
print("Accuracy = [{:.2f}]".format(Acc))
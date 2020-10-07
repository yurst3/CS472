from CS472.tools.arff import Arff
from CS472.perceptron.perceptron import PerceptronClassifier
import matplotlib.pyplot as plt
import numpy as np
import random

debug = "data/backpropagation/linsep2nonorigin.arff"
eval = "data/backpropagation/data_banknote_authentication.arff"

mat = Arff(debug, label_count=1)
np_mat = mat.data

total = np_mat.shape[0]
split = int(total * 0.7)

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode
from tqdm import tqdm
import math

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self, k_val, label_type, col_types, weight_type='inverse_distance'): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.col_types = col_types
        self.k_val = k_val
        self.label_type = label_type #Note This won't be needed until part 5
        self.weight_type = weight_type

    def fit(self,data,labels):
        """ Fit the data; run the algorithm (for this lab really just saves the data :D)
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)
        """

        self.given_data = data
        self.given_labels = labels

        return self

    def predict(self, predict_data):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        k_pred = []

        # For each data element to predict
        for i in tqdm(range(predict_data.shape[0])):

            distances = []

            # For each data element in the given data
            for j in range(self.given_data.shape[0]):

                # Euclidean distance
                distance = 0
                for k in range(predict_data.shape[1]):
                    if self.col_types[k] == 'continuous' and self.given_data[j][k] != np.nan and predict_data[i][k] != np.nan:
                        distance += (self.given_data[j][k] - predict_data[i][k])**2
                    elif self.col_types[k] == 'continuous':
                        distance += 1

                distance = math.sqrt(distance)

                # Nominal distance
                for k in range(predict_data.shape[1]):
                    if self.col_types[k] == 'nominal' and self.given_data[j][k] != np.nan and predict_data[i][k] != np.nan:
                        if self.given_data[j][k] != predict_data[i][k]:
                            distance += 1
                    if self.col_types[k] == 'nominal':
                        distance += 0.1

                if self.weight_type == 'inverse_distance':
                    distance = 1/(distance**2) if distance != 0 else 0

                distances.append(distance)

            # Find the smallest distance
            if self.weight_type == 'inverse_distance':
                # K max args
                indexes = np.argsort(distances)[-self.k_val:]
                indexes = indexes[::-1]
            else:
                # K min args
                indexes = np.argsort(distances)[:self.k_val]

            # Vote for 1 through K most likely
            preds = []
            for k in range(1, self.k_val + 2, 2):
                pred = [self.given_labels[indexes[yeet]] for yeet in range(k)]

                if self.label_type == 'nominal':
                    m = mode(pred)[0]
                else:
                    m = sum(pred)/len(pred)

                preds.append(m)

            k_pred.append(preds)

        return np.array(k_pred)

    def score(self, X, y):
            """ Return accuracy of model on a given dataset. Must implement own score function.
            Args:
                    X (array-like): A 2D numpy array with data, excluding targets
                    y (array-like): A 2D numpy array with targets
            Returns:
                    score : float
                            Mean accuracy of self.predict(X) wrt. y.
            """
            predictions = self.predict(X)

            scores = []

            for col in range(predictions.shape[1]):
                error = 0
                for row in range(predictions.shape[0]):
                    if self.label_type == 'nominal' and predictions[row, col] == y[row]:
                        error += 1
                    elif self.label_type == 'real':
                        error += (y[row] - predictions[row, col])**2

                scores.append(error / y.shape[0])

            return scores
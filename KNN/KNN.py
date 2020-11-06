import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import mode
import math

class KNNClassifier(BaseEstimator,ClassifierMixin):


    def __init__(self,columntype,weight_type='inverse_distance'): ## add parameters here
        """
        Args:
            columntype for each column tells you if continues[real] or if nominal[categoritcal].
            weight_type: inverse_distance voting or if non distance weighting. Options = ["no_weight","inverse_distance"]
        """
        self.columntype = columntype #Note This won't be needed until part 5
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

        predictions = []

        # For each data element to predict
        for i in range(predict_data.shape[0]):

            distances = []

            # For each data element in the given data
            for j in range(self.given_data.shape[0]):
                # Euclidean distance
                distance = math.sqrt(sum([(self.given_data[j][k] - predict_data[i][k])**2 for k in range(predict_data.shape[1])]))

                if self.weight_type == 'inverse_distance':
                    distance = 1/(distance**2)

                distances.append(distance)

            # Find the smallest distance
            if self.weight_type == 'inverse_distance':
                # 3 max args
                indexes = np.argsort(distances)[-3:]
            else:
                # 3 min args
                indexes = np.argsort(distances)[:3]

            # Vote for 3 most likely
            votes = [self.given_labels[index] for index in indexes]

            predictions.append(mode(votes)[0])

        return predictions

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

            correct = 0

            for i in range(len(predictions)):
                if predictions[i] == y[i]:
                    correct += 1

            return correct / y.shape[0]



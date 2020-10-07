import numpy as np
import random
import math
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified. The output of
#   get_weights must be in the same format as the example provided.

from sklearn.linear_model import Perceptron

class PerceptronClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self, lr=.1, shuffle=True, deterministic=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle: Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
        """
        self.lr = lr
        self.shuffle = shuffle
        self.deterministic = deterministic

    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
            initial_weights (array-like): allows the user to provide initial weights

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.weights = np.zeros(X.shape[1] + 1) if not initial_weights else initial_weights

        random.seed(X[0][0])

        old_accuracy = 1
        sum_d_accuracy = 0
        miss_rate = []

        # Initialize the patterns/targets
        patterns = self.add_bias(X)
        targets = y

        if self.deterministic is not None:
            epochs = self.deterministic
        else:
            epochs = 1000

        counter = 0

        # For each epoch
        for epoch in range(epochs):
            counter += 1

            # Shuffle if needed
            if self.shuffle is True:
                patterns, targets = self._shuffle_data(patterns, targets)

            ti = 0
            for pattern in patterns:

                # Calculate the activation 0 if net < 0 else 1
                activation = 0 if np.dot(pattern, self.weights.transpose()) <= 0 else 1

                # Calculate change in weights lr (t - a) x
                d_weights = np.multiply(np.subtract(targets[ti],activation), pattern.transpose()) * self.lr

                # Add to weights
                self.weights = np.add(d_weights, self.weights)

                ti += 1

            # Stopping criterion
            new_accuracy = self.score(X, y)
            miss_rate.append((1,1 - new_accuracy))
            sum_d_accuracy += math.fabs(new_accuracy - old_accuracy)

            # Every 5 epochs, check if the average change in accuracy is less than 0.01
            if counter % 5 == 0 and sum_d_accuracy / 5 < 0.01:
                # Only break if not deterministic
                if self.deterministic is None: break
            else:
                # Reset accuracy change sum to 0
                if counter % 5 == 0:
                    sum_d_accuracy = 0
                old_accuracy = new_accuracy

        return counter, miss_rate, self

    def add_bias(self, patterns):
        pattern_bias = np.array([[1] for i in range(patterns.shape[0])])

        return np.concatenate((patterns, pattern_bias), axis=1)

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """

        patterns = self.add_bias(X)
        predictions = np.array([])

        for pattern in patterns:

            prediction = np.array([0 if np.dot(pattern, self.weights.transpose()) <= 0 else 1])
            predictions = np.concatenate((predictions, prediction), axis=0)

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
        hit = 0

        ti = 0
        for prediction in predictions:
            if prediction == y[ti]:
                hit += 1
            ti += 1

        return hit / len(y)

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        matrix = np.concatenate((X, y), axis=1)
        random.shuffle(matrix)

        labels = matrix[:,-1].reshape(-1,1)
        patterns = matrix[:,:-1]

        return patterns, labels

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.weights

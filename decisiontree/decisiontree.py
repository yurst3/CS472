import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score

class DTClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,counts=None):
        """ Initialize class with chosen hyperparameters.
        Args:
        Optional Args (Args we think will make your life easier):
            counts: A list of Ints that tell you how many types of each feature there are
        Example:
            DT  = DTClassifier()
            or
            DT = DTClassifier(count = [2,3,2,2])
            Dataset = 
            [[0,1,0,0],
            [1,2,1,1],
            [0,1,1,0],
            [1,2,0,1],
            [0,0,1,1]]

        """
        self.counts = counts

    def fit(self, X, y):
        """ Fit the data; Make the Desicion tree

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets

        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        total_info = 0
        for i in range(self.counts[-1]):
            p_i = sum(np.where(y == i, 1, 0)) / y.shape[0]
            total_info += -p_i * math.log2(p_i)

        info_gains = self._calc_info_gains(X, y, total_info)

        return self

    def _calc_info_gains(self, X, y, prev_info):
        info_gains = np.zeros(len(self.counts) - 1)

        # For each attribute
        for i in range(X.shape[1]):

            # For each attribute value
            for j in range(self.counts[i]):
                values = sum(np.where(X[:, i] == j, 1, 0))
                s_j = values / X.shape[0]
                info_s_j = 0

                # For each target value
                for k in range(self.counts[-1]):
                    # Lol I am running out of good variable names
                    where = [1 if X[reee, 0] == j and y[reee, 0] == k else 0 for reee in range(X.shape[0])]
                    s_k = sum(where) / values
                    info_s_j += -s_k * math.log2(s_k) if s_k != 0 else 0

                info_gains[i] += s_j * info_s_j

            info_gains[i] = prev_info - info_gains[i]

        return info_gains

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass


    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets 
        """
        return 0


import numpy as np
import math
import numpy.random as random
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
#   * get_weights
#   They must take at least the parameters below, exactly as specified.

class MLPClassifier(BaseEstimator,ClassifierMixin):

    def __init__(self,lr=.1, momentum=0, shuffle=True,hidden_layer_widths=None):
        """ Initialize class with chosen hyperparameters.

        Args:
            lr (float): A learning rate / step size.
            shuffle(boolean): Whether to shuffle the training data each epoch. DO NOT SHUFFLE for evaluation / debug datasets.
            momentum(float): The momentum coefficent 
        Optional Args (Args we think will make your life easier):
            hidden_layer_widths (list(int)): A list of integers which defines the width of each hidden layer if hidden layer is none do twice as many hidden nodes as input nodes.
        Example:
            mlp = MLPClassifier(lr=.2,momentum=.5,shuffle=False,hidden_layer_widths = [3,3]),  <--- this will create a model with two hidden layers, both 3 nodes wide
        """
        self.hidden_layer_widths = hidden_layer_widths
        self.lr = lr
        self.momentum = momentum
        self.shuffle = shuffle


    def fit(self, X, y, initial_weights=None):
        """ Fit the data; run the algorithm and adjust the weights to find a good solution

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
            y (array-like): A 2D numpy array with the training targets
        Optional Args (Args we think will make your life easier):
            initial_weights (array-like): allows the user to provide initial weights
        Returns:
            self: this allows this to be chained, e.g. model.fit(X,y).predict(X_test)

        """
        self.num_patterns = X.shape[0]
        self.num_predictions = y.shape[1]
        self.pattern_elements = X.shape[1]
        self.initial_weights = self.initialize_weights() if not initial_weights else initial_weights

        # Stochastic weight update (update after each trained pattern)
        for i in range(self.num_patterns):
            pattern = X[i]
            target = y[i]
            outputs = [[]]

            # input --> hidden layer output calculations
            for i in range(self.hidden_layer_widths[0]):
                weights = self.initial_weights[0][i::self.hidden_layer_widths[0]]
                net = sum(pattern * weights)
                outputs[0].append(self._activation(net))

            # hidden --> hidden layer output calculations
            for i in range(1, len(self.hidden_layer_widths)):
                outputs.append([])
                for j in range(self.hidden_layer_widths[i]):
                    weights = self.initial_weights[i][j::self.hidden_layer_widths[i]]
                    net = sum(outputs[i - 1] * weights)
                    outputs[i].append(self._activation(net))

            # hidden --> output layer output calculations
            outputs.append([])
            for i in range(self.num_predictions):
                weights = self.initial_weights[-1][i::self.num_predictions]
                net = sum(outputs[-2] * weights)
                outputs[-1].append(self._activation(net))

        return self

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        pass

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        # random weight initialization (small random weights with 0 mean)
        mean = 0
        std_dev = 0.01

        weights = []

        # Generate input --> hidden weights
        num_input_weights = self.pattern_elements * self.hidden_layer_widths[0]
        #weights.append(np.zeros(num_input_weights))
        weights.append(random.normal(mean, std_dev, num_input_weights))

        # Generate hidden --> hidden weights
        for i in range(1, len(self.hidden_layer_widths)):
            num_hidden_weights = self.hidden_layer_widths[i-1] * self.hidden_layer_widths[i]
            #weights.append(np.zeros(num_hidden_weights))
            weights.append(random.normal(mean, std_dev, num_hidden_weights))

        # Generate hidden --> output weights
        #weights.append(np.zeros(self.hidden_layer_widths[-1] * self.num_predictions))
        weights.append(random.normal(mean, std_dev, self.hidden_layer_widths[-1] * self.num_predictions))

        return weights

    def score(self, X, y):
        """ Return accuracy of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array with targets

        Returns:
            score : float
                Mean accuracy of self.predict(X) wrt. y.
        """

        return 0

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        pass

    def _activation(self, net):
        return 1 / (1 + (math.e ** (-net)))

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.initial_weights

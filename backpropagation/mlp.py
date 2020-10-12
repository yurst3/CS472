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

        self.prev_weight_changes = [np.zeros(len(layer)) for layer in self.initial_weights]

        # Stochastic weight update (update after each trained pattern)
        for epoch in range(100):
            for i in range(self.num_patterns):

                # Add bias to pattern
                pattern = np.concatenate((X[i], np.array([1.])))
                target = y[i]

                # Create output container for forward/back pass
                self.outputs = [[] for i in range(len(self.initial_weights))]

                # Compute output at each node in all layers
                self._foward(pattern)

                # Compute weight change for each layer
                weight_changes = self._backward(pattern, target)

                # Apply weight changes
                self.initial_weights = [np.add(self.initial_weights[i], weight_changes[i])
                                        for i in range(len(weight_changes))]

                self.prev_weight_changes = weight_changes

        return self

    def _activation(self, net):
        return 1 / (1 + (math.e ** (-net)))

    def _derivative(self, output):
        return output * (1 - output)

    def _foward(self, pattern):
        # Assuming 2 layers, each w/ 3 nodes [3, 3]
        # Weights for input to first hidden layer will be organized as such:
        #
        # [in1->node1, in1->node2, in1->node3,
        # in2->node1, in2->node2, in2->node3,
        # bias->node1, bias->node2, bias->node3]
        #
        # Slice to calculate weights for node1 is [0::3], giving 4 total weights

        # input --> hidden layer output calculations
        for node in range(self.hidden_layer_widths[0]):
            weights = self.initial_weights[0][node::self.hidden_layer_widths[0]]
            net = np.dot(pattern, weights)
            self.outputs[0].append(self._activation(net))

        # hidden --> hidden layer output calculations
        for layer in range(1, len(self.hidden_layer_widths)):
            for node in range(self.hidden_layer_widths[layer]):
                weights = self.initial_weights[layer][node::self.hidden_layer_widths[layer]]
                # Add bias to outputs
                prev_outputs = np.concatenate((self.outputs[layer - 1], np.array([1.])))
                net = np.dot(prev_outputs, weights)
                self.outputs[layer].append(self._activation(net))

        # hidden --> output layer output calculations
        for node in range(self.num_predictions):
            weights = self.initial_weights[-1][node::self.num_predictions]
            # Add bias to outputs
            prev_outputs = np.concatenate((self.outputs[-2], np.array([1.])))
            net = np.dot(prev_outputs, weights)
            self.outputs[-1].append(self._activation(net))

    def _backward(self, pattern, target):
        weight_changes = [[] for _ in range(len(self.initial_weights))]
        prev_deltas = [np.zeros(len(out_array)) for out_array in self.outputs]

        for layer in range(1, len(self.initial_weights) + 1):
            num_nodes = len(self.outputs[-layer])
            for weight in range(len(self.initial_weights[-layer])):
                # Calculating weight change for w_i,j
                # Get output j (current layer)
                index_j = weight % num_nodes
                O_j = self.outputs[-layer][index_j]

                # Get output i (previous layer or pattern)
                if layer != len(self.initial_weights):
                    index_i = weight // num_nodes
                    O_i = self.outputs[-layer - 1][index_i] if index_i < len(self.outputs[-layer - 1]) else 1
                else:
                    index_i = weight % len(pattern)
                    O_i = pattern[index_i]

                # Calculate delta
                if layer == 1:
                    delta_j = (target[index_j] - O_j) * self._derivative(O_j)
                    prev_deltas[-layer][index_j] = delta_j
                else:
                    # Layer = -2, i->j
                    # [in0->n0, in0->n1, in0->n2, in0->n3,
                    # in1->n0, in1->n1, in1->n2, in1->n3,
                    # bias->n0, bias->n1, bias->n2, bias->n3]
                    #
                    # Layer = -1, j->k
                    # [n0->o0,
                    # n1->o0,
                    # n2->o0,
                    # n3->o0,
                    # bias->o0]

                    num_prev_nodes = len(self.outputs[-layer + 1])
                    w_jk = self.initial_weights[-layer + 1][(index_j * num_prev_nodes):(index_j * num_prev_nodes) + num_prev_nodes]
                    delta_k = prev_deltas[-layer + 1]

                    delta_j = np.dot(delta_k, w_jk) * self._derivative(O_j)
                    prev_deltas[-layer][index_j] = delta_j

                # Add momentum and append weight change to array
                weight_change = self.lr * O_i * delta_j
                weight_change += self.momentum * self.prev_weight_changes[-layer][weight]
                weight_changes[-layer].append(weight_change)

        return weight_changes

    def predict(self, X):
        """ Predict all classes for a dataset X
        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets
        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        self.num_patterns = X.shape[0]
        predictions = []

        for i in range(self.num_patterns):
            pattern = np.concatenate((X[i], np.array([1.])))

            # Create output container for forward
            self.outputs = [[] for i in range(len(self.initial_weights))]

            # Compute output at each node in all layers
            self._foward(pattern)

            predictions.append(self.outputs[-1])

        return predictions

    def initialize_weights(self):
        """ Initialize weights for perceptron. Don't forget the bias!

        Returns:

        """
        # random weight initialization (small random weights with 0 mean)
        # last weight in each array is the bias
        mean = 0
        std_dev = 5

        weights = []

        # Generate input --> hidden weights
        num_input_weights = (self.pattern_elements + 1) * self.hidden_layer_widths[0]
        #weights.append(np.zeros(num_input_weights))
        weights.append(random.normal(mean, std_dev, num_input_weights))

        # Generate hidden --> hidden weights
        for i in range(1, len(self.hidden_layer_widths)):
            num_hidden_weights = (self.hidden_layer_widths[i-1] + 1) * self.hidden_layer_widths[i]
            #weights.append(np.zeros(num_hidden_weights))
            weights.append(random.normal(mean, std_dev, num_hidden_weights))

        # Generate hidden --> output weights
        num_output_weights = (self.hidden_layer_widths[-1] + 1) * self.num_predictions
        #weights.append(np.zeros(num_output_weights))
        weights.append(random.normal(mean, std_dev, num_output_weights))

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

        predictions = self.predict(X)
        error = 0

        for i in range(y.shape[0]):
            for j in range(y.shape[1]):
                error += y[i][j] - predictions[i][j]

        return error / (y.shape[0] * y.shape[1])

    def _shuffle_data(self, X, y):
        """ Shuffle the data! This _ prefix suggests that this method should only be called internally.
            It might be easier to concatenate X & y and shuffle a single 2D array, rather than
             shuffling X and y exactly the same way, independently.
        """
        pass

    ### Not required by sk-learn but required by us for grading. Returns the weights.
    def get_weights(self):
        return self.initial_weights

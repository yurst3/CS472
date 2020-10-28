import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin
import scipy.stats as stats

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
class _Node:
    def __init__(self, attr_splits, decision, target_decision=None):
        self.attribute_splits = attr_splits
        self.decision = decision
        # None if not a leaf node
        self.target_decision = target_decision
        self.children = []

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
        self.root = _Node([], decision=None)

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
            total_info += -p_i * math.log2(p_i) if p_i != 0 else 0

        self._build_decision_tree(X, y, total_info, self.root, np.arange(X.shape[1]))

        # This is for debugging, just checks if there are any holes in the tree
        truth = self._tree_complete(self.root)

        return self

    def _build_decision_tree(self, X, y, total_info, node, og_indexes):

        # Info gain for every attribute in this current split (NOT every attribute in total)
        info_gains = self._calc_info_gains(X, y, total_info)
        info_index = np.argmax(info_gains)

        attribute_index = og_indexes[info_index]

        # Copy the current split path and append the new split
        node.attribute_splits.append(attribute_index)

        # Delete the column with the chosen attribute to split on
        new_X = np.delete(X, info_index, axis=1)

        # For each attribute value in the chosen split
        for attr_val in list(set(X[:, info_index])):
            check = [True if X[row, info_index] == attr_val else False for row in range(X.shape[0])]
            new_y = y[check]

            cur_split_copy = node.attribute_splits.copy()

            # If this is a pure node, assign a target attribute
            if max(new_y) - min(new_y) == 0:
                node.children.append(_Node(attr_splits=cur_split_copy,
                                           decision=attr_val,
                                           target_decision=new_y[0]))

            # If this isn't a pure node, keep splitting
            else:
                # Check if there are any attributes left
                if new_X.shape[1] > 0:
                    node.children.append(_Node(attr_splits=cur_split_copy,
                                               decision=attr_val))

                    # Remove all other attribute values that aren't this one
                    next_X = new_X[check]
                    next_og_indexes = np.delete(og_indexes, info_index)

                    self._build_decision_tree(X=next_X,
                                              y=new_y,
                                              total_info=total_info,
                                              node=node.children[-1],
                                              og_indexes=next_og_indexes)

                # If there aren't any attributes left to split on and this node is impure
                else:
                    # Decide based on the mode of the new Y
                    node.children.append(_Node(attr_splits=cur_split_copy,
                                               decision=attr_val,
                                               target_decision=stats.mode(new_y)[0][0]))

    def _tree_complete(self, node):

        # Base case
        if len(node.children) == 0:
            return False if node.target_decision is None else True

        # Recursion
        else:
            check = [self._tree_complete(child) for child in node.children]

            return all(check)

    def _calc_info_gains(self, X, y, total_info):
        info_gains = np.zeros(X.shape[1])

        # For each attribute
        for i in range(X.shape[1]):

            # For each attribute value
            for j in list(set(X[:, i])):
                values = sum(np.where(X[:, i] == j, 1, 0))
                s_j = abs(values) / X.shape[0]
                info_s_j = 0

                # For each target value
                for k in list(set(y[:, 0])):
                    where = [1 if X[row, i] == j and y[row, 0] == k else 0 for row in range(X.shape[0])]
                    s_k = sum(where) / values
                    info_s_j += s_k * math.log2(s_k) if s_k != 0 else 0

                info_gains[i] += s_j * -info_s_j if values > 0 else 0

            info_gains[i] = total_info - info_gains[i]

        return info_gains

    def predict(self, X):
        """ Predict all classes for a dataset X

        Args:
            X (array-like): A 2D numpy array with the training data, excluding targets

        Returns:
            array, shape (n_samples,)
                Predicted target values per element in X.
        """
        predictions = []

        for data in X:
            predictions.append(self._decide(self.root, data))

        return np.array(predictions)

    def _decide(self, node, data):

        if node.target_decision is not None:
            return node.target_decision[0]

        else:
            answer = data[node.attribute_splits[-1]]
            for child in node.children:
                if child.decision == answer:
                    return self._decide(child, data)

    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets 
        """
        predictions = self.predict(X)

        correct = 0

        for i in range(y.shape[0]):
            if y[i] == predictions[i]:
                correct += 1

        return correct / y.shape[0]


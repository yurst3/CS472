import numpy as np
import math
from sklearn.base import BaseEstimator, ClassifierMixin

### NOTE: The only methods you are required to have are:
#   * predict
#   * fit
#   * score
class _Node:
    def __init__(self, attr_splits, decision=None):
        self.attribute_splits = attr_splits
        self.decision = decision
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
        self.root = _Node([])

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

        self._build_decision_tree(X, y, total_info, self.root)

        return self

    def _build_decision_tree(self, X, y, total_info, node):

        # Info gain for every attribute in this current split (NOT every attribute in total)
        info_gains = self._calc_info_gains(X, y, total_info)

        # If this is the root node
        if len(node.attribute_splits) == 0:
            attribute_index = np.argmax(info_gains)

        # If this is not the root node
        else:
            # Convert to index that plays nice with self.count
            attribute_index = np.argmin(info_gains)
            for split in node.attribute_splits:
                if attribute_index > split:
                    attribute_index += 1

        # Copy the current split path and append the new split
        cur_split_copy = node.attribute_splits.copy()
        cur_split_copy.append(attribute_index)

        # Delete the column with the chosen attribute to split on
        new_X = np.delete(X, attribute_index, axis=1)

        # For each attribute value in the chosen split
        for attr_val in range(self.counts[attribute_index]):
            where = np.array([y[row, 0] if X[row, attribute_index] == attr_val else np.NaN for row in range(X.shape[0])])
            where = where[~np.isnan(where)]

            # If this is a pure node, assign a target attribute
            if max(where) - min(where) == 0:
                node.children.append(_Node(attr_splits=cur_split_copy,
                                           decision=where[0]))

            # If this isn't a pure node, keep splitting
            else:
                check = [True if X[row, attribute_index] == attr_val else False for row in range(X.shape[0])]
                new_y = y[check]

                # Check if there are any attributes left
                if new_X.shape[0] > 0:
                    node.children.append(_Node(attr_splits=cur_split_copy))

                    # Remove all other attribute values that aren't this one
                    new_X = new_X[check]

                    self._build_decision_tree(X=new_X,
                                              y=new_y,
                                              total_info=(total_info - max(info_gains)),
                                              node=node.children[-1])

                # If there aren't any attributes left to split on and this node is impure
                else:
                    # Take decide based on the first target attribute
                    node.children.append(_Node(attr_splits=cur_split_copy, decision=new_y[0]))


    def _tree_complete(self, node):

        # Base case
        if len(node.children) == 0:
            return False if node.decision is None else True

        # Recursion
        else:
            check = [self._tree_complete(child) for child in node.children]

            return all(check)


    def _calc_info_gains(self, X, y, total_info):
        info_gains = np.zeros(X.shape[1])

        # For each attribute
        for i in range(X.shape[1]):

            # For each attribute value
            for j in range(int(max(X[:,i]) - min(X[:,i])) + 1):
                values = sum(np.where(X[:, i] == j, 1, 0))
                s_j = abs(values) / X.shape[0]
                info_s_j = 0

                # For each target value
                for k in range(int(max(y[:,0]) - min(y[:,0])) + 1):
                    where = [1 if X[row, 0] == j and y[row, 0] == k else 0 for row in range(X.shape[0])]
                    s_k = sum(where) / values
                    info_s_j += -s_k * math.log2(s_k) if s_k != 0 else 0

                info_gains[i] += s_j * info_s_j

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
        pass


    def score(self, X, y):
        """ Return accuracy(Classification Acc) of model on a given dataset. Must implement own score function.

        Args:
            X (array-like): A 2D numpy array with data, excluding targets
            y (array-like): A 2D numpy array of the targets 
        """
        return 0


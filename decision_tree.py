import numpy as np
from collections import Counter
from treenode import TreeNode


class DecisionTree():
    """
    Decision Tree Classifier
    Training: Use "train" function with train set features and labels
    Predicting: Use "predict" function with test set features
    """

    def __init__(self, max_depth=4, min_samples_leaf=1) -> None:
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.tree_info = {}

    def entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]
    
    def data_entropy(self, labels: list) -> float:
        return self.entropy(self.class_probabilities(labels))
    
    def partition_entropy(self, subsets: list) -> float:
        """
            subsets = list of label lists ( EX: [[1,0,0], [1,1,1], [0,0,1,0,0]] )
        """
        total_count = sum([len(subset) for subset in subsets])
        return sum([self.data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2
        
    def find_best_split(self, data: np.array) -> tuple:
        """
        Finds the best split (with the lowest entropy) given data
        Returns 2 splitted groups
        """
        min_entropy = 1e6
        min_entropy_feature_idx = None
        min_entropy_feature_val = None

        for idx in range(data.shape[1]-1):
            feature_val = np.median(data[:, idx])
            g1, g2 = self.split(data, idx, feature_val)
            entropy = self.partition_entropy([g1[:, -1], g2[:, -1]])
            if entropy < min_entropy:
                min_entropy = entropy
                min_entropy_feature_idx = idx
                min_entropy_feature_val = feature_val
                g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val

    def find_label_probs(self, data: np.array) -> np.array:

        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)

        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def create_tree(self, data: np.array, current_depth: int) -> TreeNode:
        
        # Check if the max depth has been reached
        if current_depth >= self.max_depth:
            return None
        
        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val = self.find_best_split(data)
        
        # Find label probs for the node
        label_probabilities = self.find_label_probs(data)

        # Create node
        node = TreeNode(data, split_feature_idx, split_feature_val, label_probabilities)

        # Check if the min_samples_leaf has been satisfied
        current_depth += 1
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node        

        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)
        
        return node
    
    def predict_one_sample(self, X: np.array) -> np.array:
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        
        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        # Initialize the tree information
        self.tree_info['depth'] = 0

        self.tree = self.create_tree(data=train_data, current_depth=0)

    def predict_proba(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """Returns the predicted probs for a given data set"""

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds    
        
    def print_recursive(self, node: TreeNode, level=0) -> None:
        if node != None:
            self.print_recursive(node.left, level + 1)
            unique_values, value_counts = np.unique(node.data[:,-1], return_counts=True)
            output = ", ".join([f"{value}->{count}" for value, count in zip(unique_values, value_counts)])

            if (node.left and node.right):
                print('    ' * 4 * level + '-> ' \
                    + ' Node: ' + str(node.node_def) + ' ' )
            else:
                print('    ' * 4 * level + '-> ' \
                      + ' LEAF: '
                        + ' Labels Count=' + output \
                            + ' Pred Probs=' + str(node.prediction_probs))                
                                
            self.print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self.print_recursive(node=self.tree)
import numpy as np
from collections import Counter
from treenode import TreeNode


class DecisionTree():

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
    
    def split(self, data, feature_idx, feature_val):
        
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2
        
    def find_best_split(self, data):
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

    def create_tree(self, data, current_depth):
        
        # Check if the max depth has been reached
        if current_depth >= self.max_depth:
            return None
        
        # Create the node
        split_1_data, split_2_data, split_feature_idx, split_feature_val = self.find_best_split(data)
        node_prediction = Counter(list(data[:,-1])).most_common(1)[0][0]
        node = TreeNode(data, split_feature_idx, split_feature_val, node_prediction)
        current_depth += 1

        # Check if the min_samples_leaf has been satisfied
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node        

        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)
        
        return node
    
    def predict_one_sample(self, X):
        """Returns prediction for 1 dim array"""
        node = self.tree

        # Finds the leaf which X belongs
        while node:
            
            pred = node.prediction
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred

    def train(self, X_train, Y_train):
        
        # Concat features and labels
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)

        # Initialize the tree information
        self.tree_info['depth'] = 0

        self.tree = self.create_tree(data=train_data, current_depth=0)

    def predict(self, X_set):
        """Returns the predictions for a given data set"""

        predictions = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        
        return predictions

    def print_recursive(self, node, level=0):
        if node != None:
            self.print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '-> ' \
                  + ' Idx=' + str(node.feature_idx) + ' ' \
                    + ' Val=' + str(round(node.feature_val, 2))\
                        + ' Labels=' + str(np.unique(node.data[:,-1], return_counts=True))\
                        + ' Pred=' + str(node.prediction)
                        )
            self.print_recursive(node.right, level + 1)

    def print_tree(self):
        self.print_recursive(self.tree)
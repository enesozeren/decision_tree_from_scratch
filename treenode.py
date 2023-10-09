class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction = prediction
        self.left = None
        self.right = None
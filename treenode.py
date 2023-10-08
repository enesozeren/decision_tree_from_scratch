class TreeNode():
    def __init__(self, data, feature_idx, feature_val) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.left = None
        self.right = None
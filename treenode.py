class TreeNode():
    def __init__(self, data, feature_idx, feature_val, prediction_probs) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.node_def = f"Is X[{feature_idx}] < {feature_val}"
        self.left = None
        self.right = None
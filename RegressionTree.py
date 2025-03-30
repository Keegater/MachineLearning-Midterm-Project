import numpy as np


class RegressionTree:
    def __init__(self, data, max_height=None, min_leaf_size=None, limit_type="height"):

        self.max_height = max_height            #maximum tree height (number of branches)
        self.min_leaf_size = min_leaf_size      #minimum number of samples in a leaf (partition)
        self.limit_type = limit_type            #limit tree growth by "height", "leaf_size" or neither

        # Number of features, assuming the last column is the target variable
        self.n_features = data.shape[1] - 1

        # Build the tree recursively
        self.tree = self._build_tree(data, height=0)


    def _build_tree(self, data, height):
        """
        Recursively builds the regression tree.

        Returns a dictionary which represents the node.
              Leaf nodes only have a "value" key
              Branch nodes have keys "feature", "threshold", "left", "right", and "value" (mean value of node)
        """
        X = data[:, :self.n_features]
        y = data[:, -1]
        node_value = np.mean(y)

        # If only one sample, create a leaf node
        if data.shape[0] <= 1:
            return {"value": node_value}

        # Check for stopping criteria
        if self.limit_type == "height" and self.max_height is not None and height >= self.max_height:
            return {"value": node_value}
        if self.limit_type == "leaf_size" and self.min_leaf_size is not None and data.shape[0] <= self.min_leaf_size:
            return {"value": node_value}
        # If features have zero variance, no useful split exists
        if np.all(np.std(X, axis=0) == 0):
            return {"value": node_value}

        # Compute current sum of squared errors (SSE) for this node
        current_error = np.sum((y - node_value) ** 2)

        # Initialize best split variables
        best_feature = None
        best_threshold = None
        best_error_reduction = 0
        best_left = None
        best_right = None

        # Search over all features for the best split
        for feature in range(self.n_features):
            # Get all unique sorted values of the current feature
            values = np.unique(X[:, feature])
            # If there is only one unique value, then no split is possible on this feature
            if len(values) == 1:
                continue
            # Use midpoints between consecutive unique values as candidate thresholds
            candidate_thresholds = (values[:-1] + values[1:]) / 2.0

            for threshold in candidate_thresholds:
                # Split the data based on the candidate threshold
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask  # equivalent to X[:, feature] >= threshold

                # Skip split if one side is empty
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue

                left_data = data[left_mask]
                right_data = data[right_mask]

                # Calculate the error (sum of squared errors) for the left split
                left_y = left_data[:, -1]
                left_mean = np.mean(left_y)
                error_left = np.sum((left_y - left_mean) ** 2)

                # Calculate the error for the right split
                right_y = right_data[:, -1]
                right_mean = np.mean(right_y)
                error_right = np.sum((right_y - right_mean) ** 2)

                total_error = error_left + error_right
                error_reduction = current_error - total_error

                if error_reduction > best_error_reduction:
                    best_error_reduction = error_reduction
                    best_feature = feature
                    best_threshold = threshold
                    best_left = left_data
                    best_right = right_data

        # If no valid split found, make a leaf node.
        if best_feature is None:
            return {"value": node_value}

        # Otherwise, create an internal node with the best split found.
        node = {
            "feature": best_feature,
            "threshold": best_threshold,
            "value": node_value,
            "left": self._build_tree(best_left, height + 1),
            "right": self._build_tree(best_right, height + 1)
        }
        return node


    def predict(self, sample):
        """
        Predicts the target value for a given sample.

        Parameters:
            sample (np.ndarray): A 1D array of feature values.

        Returns:
            float: The predicted value (mean from the leaf node reached).
        """
        node = self.tree
        # Traverse the tree until a leaf is reached
        while "feature" in node:
            feature = node["feature"]
            threshold = node["threshold"]
            if sample[feature] < threshold:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]


    def decision_path(self, sample):
        """
        Returns the decision path as a list of strings representing the deduction rules.

        Parameters:
            sample (np.ndarray): A 1D array of feature values.

        Returns:
            list of str: Each string describes a rule followed to reach the leaf.
        """
        path = []
        node = self.tree
        while "feature" in node:
            feature = node["feature"]
            threshold = node["threshold"]
            if sample[feature] < threshold:
                path.append(f"If feature {feature} < {threshold:.3f}, go left")
                node = node["left"]
            else:
                path.append(f"If feature {feature} >= {threshold:.3f}, go right")
                node = node["right"]
        path.append(f"Predict {node['value']:.3f}")
        return path

# Example usage:
if __name__ == "__main__":
    # Create a small synthetic dataset.
    # Each row: [feature1, feature2, ..., target]
    data = np.array([
        [2.5, 3.0, 10.0],
        [3.5, 2.0, 12.0],
        [1.5, 4.0, 9.0],
        [3.0, 3.5, 11.0],
        [2.0, 2.5, 8.5]
    ])

    # Build the tree using maximum height as the stopping criterion.
    tree = RegressionTree(data, max_height=3, min_leaf_size=2, limit_type="height")

    # Predict for a new sample
    sample = np.array([3.0, 3.0])
    prediction = tree.predict(sample)
    print("Prediction:", prediction)

    # Show the decision path
    path = tree.decision_path(sample)
    print("Decision path:")
    for rule in path:
        print(rule)
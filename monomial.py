import numpy as np


class Monomial:
    def __init__(self):
        self.features_splits = {}
        self.features = None
        self.thresholds = None

    def add_split(self, split_feature, split_value):
        if split_feature in self.features_splits:
            self.features_splits[split_feature] = max(self.features_splits[split_feature], split_value)
        else:
            self.features_splits[split_feature] = split_value

    def __eq__(self, other):
        if set(self.features_splits.keys()) != set(other.features_splits.keys()):
            return False
        for feature, threshold in self.features_splits.items():
            if not np.isclose(threshold, other.features_splits[feature]):
                return False
        return True

    def finalize(self):
        feature_items = list(self.features_splits.items())
        self.features = np.array([feature_item[0] for feature_item in feature_items], dtype=np.int)
        self.thresholds = np.array([feature_item[1] for feature_item in feature_items])

    def __call__(self, x):
        if self.features is None:
            self.finalize()
        x = np.array(x)
        return 1 if np.all(np.less_equal(self.thresholds, x[self.features])) else 0

    def print_to_console(self):
        if len(self.features_splits) == 0:
            print(1, end='')
        for feature, threshold in self.features_splits.items():
            print(f'[x[{feature}] >= {threshold}]', end='')
        print()

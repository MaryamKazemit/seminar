from sklearn.base import clone
from sklearn.cluster import AgglomerativeClustering
from collections import Counter


class SelfAdjustedActiveLearning:
    def __init__(self, base_classifier, n_dynamic_classifiers, block_size, threshold):
        self.base_classifier = base_classifier
        self.n_dynamic_classifiers = n_dynamic_classifiers
        self.block_size = block_size
        self.threshold = threshold
        self.dynamic_classifiers = []
        self.stable_classifier = None
        self.data_buffer = []
        self.uncertainty_threshold = threshold

    def adjust_uncertainty_threshold(self, IR):
        # Adjust the uncertainty threshold based on imbalance ratio (IR)
        self.uncertainty_threshold *= (1 - IR / 2)

    def train_classifier(self, X, y):
        # Train a new classifier with the given data
        classifier = clone(self.base_classifier)
        classifier.fit(X, y)
        return classifier

    def cluster_based_initialization(self, X):
        # Initialize using cluster-based method
        clustering = AgglomerativeClustering(
            n_clusters=self.n_dynamic_classifiers)
        labels = clustering.fit_predict(X)
        return labels

    def calculate_imbalance_ratio(labels):
        counter = Counter(labels)
        if len(counter) != 2:
            raise ValueError(
                "The function is designed for binary classification only.")

        # Get the count of instances for each class
        count_class_0 = counter[0]
        count_class_1 = counter[1]

        # Calculate the imbalance ratio
        if count_class_0 == 0 or count_class_1 == 0:
            raise ValueError("One of the classes has no instances.")

        imbalance_ratio = min(count_class_0, count_class_1) / \
            max(count_class_0, count_class_1)

        return imbalance_ratio

    def process_instance(self, x):
        # Process a single instance and decide whether to query for label
        if self.stable_classifier is not None and len(self.dynamic_classifiers) > 0:
            stable_pred = self.stable_classifier.predict([x])
            dynamic_preds = [clf.predict([x])
                             for clf in self.dynamic_classifiers]

            # Check if the predictions are consistent and whether to query the label
            if stable_pred in dynamic_preds and dynamic_preds.count(stable_pred) > len(dynamic_preds) / 2:
                return False  # No need to query
            return True  # Need to query
        return True  # Default to query if no classifiers are available

    def fit(self, stream_data):
        # Fit the model with the stream of data
        for i, (x, y) in enumerate(stream_data):
            if len(self.data_buffer) < self.block_size:
                if self.process_instance(x):
                    self.data_buffer.append((x, y))
            else:
                # Train a new classifier when the block is full
                X, y = zip(*self.data_buffer)
                if self.stable_classifier is None:
                    self.stable_classifier = self.train_classifier(X, y)
                else:
                    # Cluster-based initialization for dynamic classifiers
                    labels = self.cluster_based_initialization(X)
                    for label in set(labels):
                        indices = [i for i, l in enumerate(
                            labels) if l == label]
                        X_subset, y_subset = X[indices], y[indices]
                        new_classifier = self.train_classifier(
                            X_subset, y_subset)
                        self.dynamic_classifiers.append(new_classifier)
                        if len(self.dynamic_classifiers) > self.n_dynamic_classifiers:
                            # Remove the oldest classifier
                            self.dynamic_classifiers.pop(0)

                # Adjust the threshold based on the current imbalance
                # Implement this function based on your needs
                IR = calculate_imbalance_ratio(y)
                self.adjust_uncertainty_threshold(IR)

                self.data_buffer = []  # Clear the buffer for the next block

# Usage example but how and where????:
# base_classifier = naiive bayes write the code
# algorithm = SelfAdjustedActiveLearning(base_classifier, n_dynamic_classifiers=5, block_size=100, threshold=0.5)
# algorithm.fit(stream_data)

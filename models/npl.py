"""
Concept similar to : https://www.analyticsvidhya.com/blog/2017/09/pseudo-labelling-semi-supervised-learning-technique/
"""
import numpy as np
from dataload import load_diabetes_data
from utils import matrix2vec, eval
import sklearn.svm as svm


class confi():

    def __init__(self, model, unlabelled_data, sample_rate=0.4, upper_threshold=0.8, lower_threshold=0.2,
                 verbose=False):
        self.sample_rate = sample_rate
        self.model = model
        self.unlabelled_data = unlabelled_data
        self.verbose = verbose
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold

        # create a list of all the indices
        self.unlabelled_indices = list(range(unlabelled_data.shape[0]))

        # Number of rows to sample in each iteration
        self.sample_size = int(unlabelled_data.shape[0] * self.sample_rate)

        # Shuffle the indices
        np.random.shuffle(self.unlabelled_indices)

    def __pop_rows(self):
        """
        Function to sample indices without replacement
        """
        chosen_rows = self.unlabelled_indices[:self.sample_size]

        # Remove the chosen rows from the list of indicies (We are sampling w/o replacement)
        self.unlabelled_indices = self.unlabelled_indices[self.sample_size:]
        return chosen_rows

    def fit(self, X, y):
        """
        Perform pseudo labelling

        X: train features
        y: train targets

        """
        num_iters = int(len(self.unlabelled_indices) / self.sample_size)

        for _ in (range(num_iters) if self.verbose else range(num_iters)):
            # Get the samples
            chosen_rows = self.__pop_rows()

            # Fit to data
            self.model.fit(X, y.ravel())

            chosen_unlabelled_rows = self.unlabelled_data[chosen_rows, :]
            pseudo_labels_prob = self.model.predict_proba(chosen_unlabelled_rows)

            label_probability = np.max(pseudo_labels_prob, axis=1)
            labels_within_threshold = \
            np.where((label_probability < self.lower_threshold) | (label_probability > self.upper_threshold))[0]

            # Use argmax to find the class with the highest probability
            pseudo_labels = np.argmax(pseudo_labels_prob[labels_within_threshold], axis=1)
            chosen_unlabelled_rows = chosen_unlabelled_rows[labels_within_threshold]

            X = np.vstack((chosen_unlabelled_rows, X))
            y = np.vstack((pseudo_labels.reshape(-1, 1), np.array(y).reshape(-1, 1)))

            # Shuffle
            indices = list(range(X.shape[0]))
            np.random.shuffle(indices)

            X = X[indices]
            y = y[indices]

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def decision_function(self, X):
        return self.model.decision_function(X)


if __name__ == '__main__':
    x_label, y_label, x_unlab, x_test, y_test, _, _ = load_diabetes_data(0.5)

    svm = svm.SVC(C=1.5, kernel='rbf', probability=True)

    confisvm = confi(svm, x_unlab, sample_rate=0.4, verbose=True)

    y_label = matrix2vec(y_label)
    confisvm.fit(x_label, y_label)

    y_test_proba = confisvm.predict_proba(x_test)

    print(eval('acc', y_test, y_test_proba))

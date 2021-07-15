import numpy as np


class MNB:  # Multinomial Nb is ussed because of high accuracy.

    def __init__(self, alpha=1):  # alfa is set to 1. General info is documented in the report
        self.alpha = alpha  # alpha is 1 is called Laplace smoothing

    def fit(self, X_train, Y_train):
        m, n = X_train.shape
        self._classes = np.unique(Y_train)  # this assigning will be used for bonus part in here it is just pos-neg
        n_classes = len(self._classes)  # len is 2 but also written for calculation of the category labels

        self._priors = np.zeros(n_classes)
        self._likelihoods = np.zeros((n_classes, n))

        for idx, c in enumerate(self._classes):
            X_train_c = X_train[c == Y_train]
            self._priors[idx] = X_train_c.shape[0] / m
            self._likelihoods[idx, :] = ((X_train_c.sum(axis=0)) + self.alpha) / (
                np.sum(X_train_c.sum(axis=0) + self.alpha))

    def predict(self, X_test):
        predictions = []
        for x_test in X_test:
            posteriors = []  # append all the posteriors to here then find the max
            for i in range(len(self._classes)):  # just one row each time
                prior = np.log(self._priors[i])
                likelihoods = np.log(self._likelihoods[i, :]) * x_test  # log calculations are done to avoid underflow
                likelihood_sum = np.sum(likelihoods)
                posterior = likelihood_sum + prior
                posteriors.append(posterior)
                # posterior = likelihood x prior / evidence
            predictions.append(
                self._classes[np.argmax(posteriors)])  # argmax returns the index of the max number in the array
        return predictions

    def load_data(self, f="all_sentiment_shuffled.txt", six_category=False):
        data_set = []
        with open(f, "r", encoding="utf8") as f:
            for f2 in f:
                x = f2.split(" ", 3)  # 3 is the txt
                category_label = x[0]
                sentiment_y_label = x[1]
                doc_identifier = x[2]
                doc_tokens = x[3].split("\n")[0]  # this is just for ignoring the newline
                if not six_category:  # for negative or positve
                    data_set.append([doc_tokens, sentiment_y_label])
                else:  # for six_category classifier
                    data_set.append([doc_tokens, category_label])
        return np.array(data_set)

    def accuracy_calculator(self, prediction, real):
        N = len(real)
        counter = 1
        for i in range(N):
            if prediction[i] == real[i]:
                counter += 1
        return (counter / N) * 100


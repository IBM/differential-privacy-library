import warnings

import numpy as np
from scipy.optimize import minimize
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from diffprivlib.mechanisms import Vector


# noinspection PyPep8Naming
class LogisticRegression(BaseEstimator):
    def __init__(self, epsilon, lam=0.01, verbose=0):
        self.epsilon = epsilon
        self.verbose = verbose
        self.lam = lam
        self.beta = None

    def decision_function(self, X):
        pass

    def predict_proba(self, X):
        if self.beta is None:
            raise NotFittedError("Model not fitted. Call fit() first.")

        return np.exp(- self.loss(self.beta, X, 1))

    @staticmethod
    def loss(beta, x, label):
        # TODO: Allow for array-valued x to return losses for multiple inputs
        exponent = beta[0] + np.dot(beta[1:], x)
        return np.log(1 + np.exp(exponent)) - label * exponent

    def fit(self, X, y, sample_weight=None):
        del sample_weight

        max_norm = np.linalg.norm(X, axis=1).max()
        if max_norm > 1:
            warnings.warn("Differential privacy is only guaranteed for data whose rows have a 2-norm of at most 1. "
                          "Translate and/or scale the data accordingly to ensure differential privacy is achieved.",
                          RuntimeWarning)

        n, d = X.shape
        beta0 = np.zeros(d + 1)

        def objective(beta):
            total = 0

            for i in range(n):
                total += self.loss(beta, X[i, :], y[i])

            total /= n
            total += self.lam / 2 * np.linalg.norm(beta) ** 2

            return total

        vector_mech = Vector().set_dimensions(d + 1, n).set_epsilon(self.epsilon).set_lambda(self.lam).\
            set_sensitivity(0.25)
        noisy_objective = vector_mech.randomise(objective)

        noisy_beta = minimize(noisy_objective, beta0, method='Nelder-Mead').x
        self.beta = noisy_beta

        return self

    def predict(self, X):
        return np.int(self.predict_proba(X) > 0.5)

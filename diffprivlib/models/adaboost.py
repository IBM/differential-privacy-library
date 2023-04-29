import math
# import warnings

import numpy as np
from scipy.special import xlogy
from sklearn import ensemble
# from sklearn.base import is_regressor
# from sklearn.utils import check_random_state
# from sklearn.utils.validation import _check_sample_weight

from diffprivlib.accountant import BudgetAccountant
# from diffprivlib.mechanisms import Gaussian
# from diffprivlib.utils import PrivacyLeakWarning, warn_unused_args, check_random_state
from diffprivlib.validation import DiffprivlibMixin


class DPAdaBoostClassifier(ensemble.AdaBoostClassifier, DiffprivlibMixin):
    r"""The 'Strong Halfspace Learner' (aka HS-StL) classifier by Bun et. al (2020).
    Link: https://arxiv.org/pdf/2002.01100.pdf

    TODO

    Attributes
    ----------
    TODO

    Examples
    --------
    >>> TODO

    See also
    --------
        TODO: point to the Gaussian mechanism, and the original AdaBoostClassifier

    References
    ----------
    .. [TODO]
    """

    def __init__(self, estimator,
                 tau,  # e.g., the difference between the cov of the classes
                 k,  # e.g., I guess this'd be the avg of the cov of the classes
                 random_state=None,
                 alpha=.95, beta=0.05, epsilon=0.1, delta=0.01,
                 accountant=None, learning_rate=1.0,
                 *args, **unused_args):
        
        # A: for docs purposes - these are what all the vars mean
        desired_final_accuracy = self.alpha = alpha
        probability_of_learning_failure = self.beta = beta
        desired_final_privacy = self.epsilon = epsilon
        desired_final_privacy_approx = self.delta = delta
        density_of_lazybregboost = self.k = k
        margin_of_halfspace = self.tau = tau
        # parameters for the mechanism - will be set to Gaussian as default for now
        self.epsilon = epsilon
        self.delta = delta

        # B: configure more params of the boosting algorithm
        self.T = n_estimators = int(math.ceil(1024 * math.log10(1 / self.k)) / (self.tau ** 2))

        print(f"T: {self.T}")

        super().__init__(estimator, n_estimators=n_estimators,
                         algorithm='SAMME', random_state=random_state,
                         learning_rate=learning_rate,
                         *args, **unused_args)
        
        # C: add DP-specific fields
        self.accountant = BudgetAccountant(epsilon=self.epsilon, delta=self.delta)
        ##################
        """
        # from Theorem 27 in the paper,
        # "c" is a constant that is utilized
        # in this model to characterize the likelihood our 
        # weak learner is able to meet a "threshold" 
        # advantage lower-bound. 
        # On the advice of Bun (one of the authors), we can calculate
        # this using
        # the tail bounds for a univariate normal, via 
        # Chebyshev's inequality --> i.e., 
        # since we have a Z-distribution and (let's say) k = 3.
        """
        threshold_value_of_std_deviations = 3
        REDUNDANCY_SCALE_FACTOR = self.c = (1 / (threshold_value_of_std_deviations ** 2)) 
        #################
        # TODO: refactor magic numbers
        self.sigma = (
            (self.tau / (8 * self.c)) * math.sqrt(
                math.log10((3072 * math.log10(1 / self.k)) / (self.beta * (self.tau ** 2)))
            )
        )

    def _boost_discrete(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the 'Lazy-Bregman Next Measure (LB-NxM)' algorithm."""
        estimator = self._make_estimator(random_state=random_state)

        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, "classes_", None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)

        # Instances incorrectly classified
        incorrect = y_predict != y

        # Error fraction
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        # Stop if classification is perfect
        if estimator_error <= 0:
            return sample_weight, 1.0, 0.0

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1.0 / (n_classes - 1), 1.0])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba = y_predict_proba  # alias for readability
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)

        # Boost weight using multi-class AdaBoost SAMME alg
        estimator_weight = self.learning_rate * (
            np.log((1.0 - estimator_error) / estimator_error) + np.log(n_classes - 1.0)
        )

        print(f'weight shape: {estimator_weight.shape}')

        # Only boost the weights if it will fit again
        if not iboost == self.n_estimators - 1:
            loss = 1 - 0.5 * (np.linalg.norm(y - y_predict, 1))
            sample_weight *= np.exp(-1 * self.learning_rate * np.sum(loss))
            sample_weight_all_prod = np.prod(sample_weight)
            sample_weight = np.repeat(sample_weight_all_prod, X.shape[0])

        # Change[Zain] - go from returning 1, to the actual weight
        # return sample_weight, 1.0, estimator_error
        return sample_weight, estimator_weight, estimator_error
    
    def _boost(self, iboost, X, y, sample_weight, random_state):
        """
        Behaves actually as in the parent class, except we are 
        always going to execute the 'LB-NxM' algorithm.

        Parent class implementation: https://github.com/scikit-learn/scikit-learn/blob/364c77e047ca08a95862becf40a04fe9d4cd2c98/sklearn/ensemble/_weight_boosting.py#L529
        """
        if self.algorithm == "SAMME.R":
            self.k = self.alpha / 4  # this comes from the last line of "Algorithm 5" in the paper
            sample_weight = np.repeat(self.k, X.shape[0])
            return self._boost_real(iboost, X, y, sample_weight, random_state)

        elif self.algorithm == "SAMME":
            self.k = self.alpha / 4  # this comes from the last line of "Algorithm 5" in the paper
            sample_weight = np.repeat(self.k, X.shape[0])
            return self._boost_discrete(iboost, X, y, sample_weight, random_state)
        
        # else:  # elif self.algorithm == "LB-NxM":
        #     self.k = self.alpha / 4  # this comes from the last line of "Algorithm 5" in the paper
        #     sample_weight = np.repeat(self.k, X.shape[0])
        #     return self._boost_lb_nxm(iboost, X, y, sample_weight, random_state)
    
    def fit(self, X, y, sample_weight=None):
        '''Just like the regular, but tracks the privacy budget.'''
        self.accountant.check(self.epsilon, 0)
        fitted_self = super().fit(X, y, sample_weight=sample_weight)
        self.accountant.spend(self.epsilon, 0)
        return fitted_self

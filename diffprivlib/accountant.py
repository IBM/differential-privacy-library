# MIT License
#
# Copyright (C) IBM Corporation 2020
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Privacy budget accountant for differential privacy
"""
from numbers import Integral

import numpy as np

from diffprivlib.utils import check_epsilon_delta, BudgetError


def check_accountant(accountant):
    if accountant is not None and not isinstance(accountant, BudgetAccountant):
        raise TypeError("Accounant must be of type BudgetAccountant, got {}.".format(type(accountant)))


class BudgetAccountant:
    """Privacy budget accountant for differential privacy.

    This class creates a privacy budget accountant to track privacy spend across queries and other data accesses. Once
    initialised, the BudgetAccountant stores each privacy spend and iteratively updates the total budget spend, raising
    an error when the total budget (if specified) is exceeded. The accountant can be initialised without any maximum
    budget, to enable users track the total privacy spend of their actions without hindrance.

    Implements the accountant rules as given in [KOV17]_.

    Parameters
    ----------
    epsilon : float, default: infinity
        Target epsilon budget for the accountant.

    delta : float, default : 1
        Target delta budget for the accountant.

    slack : float, default : 0
        Slack allowed in delta for composition. Greater slack gives a smaller epsilon composition, but degrades the
        privacy guarantee offered by delta.

    spent_budget : list of tuples of the form (epsilon, delta), default : None
        List of tuples of pre-existing budget spends. Allows for a new accountant to be initialised with spends
        extracted from a previous instance.

    Attributes
    ----------
    slack : float
        The accountant's slack. Can be modified at runtime, subject to the privacy budget not being exceeded.

    spent_budget : list of tuples of the form (epsilon, delta)
        The list of privacy spends recorded by the accountant. Can be used in the initialisation of a new accountant.

    References
    ----------
    .. [KOV17] Kairouz, Peter, Sewoong Oh, and Pramod Viswanath. "The composition theorem for differential privacy."
        IEEE Transactions on Information Theory 63.6 (2017): 4037-4049.

    """
    def __init__(self, epsilon=float("inf"), delta=1, slack=0, spent_budget=None):
        check_epsilon_delta(epsilon, delta)
        self.epsilon = epsilon
        self.delta = delta
        self.__spent_budget = []
        self.slack = slack

        if spent_budget is not None:
            try:
                for _epsilon, _delta in spent_budget:
                    self.spend(_epsilon, _delta)
            except BudgetError:
                raise
            except Exception as exc:
                raise ValueError("spent_budget must be a list of tuples, of the form (epsilon, delta)") from exc

    @property
    def slack(self):
        """Slack parameter for composition.
        """
        return self.__slack

    @slack.setter
    def slack(self, slack):
        if not 0 <= slack <= 1:
            raise ValueError("Slack must be between 0 and 1 (inclusive), got {}.".format(slack))
        if slack > self.delta:
            raise ValueError("Slack must not be greater than the total delta budget ({}). "
                             "Got {}.".format(self.delta, slack))

        epsilon_spent, delta_spent = self.total_spent(slack=slack)

        if self.epsilon < epsilon_spent or self.delta < delta_spent:
            raise BudgetError("Privacy budget will be exceeded by changing slack to {}.".format(slack))

        self.__slack = slack

    @property
    def spent_budget(self):
        """List of tuples of the form (epsilon, delta) of spent privacy budget.
        """
        return self.__spent_budget

    def total_spent(self, spent_budget=None, slack=None):
        """Returns the total current privacy spend.

        `spent_budget` and `slack` can be specified as parameters, otherwise the class values will be used.

        Parameters
        ----------
        spent_budget : list of tuples of the form (epsilon, delta), default: None
            List of tuples of budget spends. If not provided, the accountant's spends will be used.

        slack : float, default : None
            Slack in delta for composition. If not provided, the accountant's slack will be used.

        Returns
        -------
        epsilon : float
            Total epsilon spend.

        delta : float
            Total delta spend.
        """
        if spent_budget is None:
            spent_budget = self.__spent_budget
        else:
            for epsilon, delta in spent_budget:
                check_epsilon_delta(epsilon, delta)

        if slack is None:
            slack = self.slack
        elif not 0 <= slack <= 1:
            raise ValueError("Slack must be between 0 and 1 (inclusive), got {}".format(slack))

        epsilon_sum, epsilon_exp_sum, epsilon_sq_sum = 0, 0, 0

        for epsilon, _ in spent_budget:
            epsilon_sum += epsilon
            epsilon_exp_sum += (1 - np.exp(-epsilon)) * epsilon / (1 + np.exp(-epsilon))
            epsilon_sq_sum += epsilon ** 2

        total_epsilon_naive = epsilon_sum
        total_delta = self.__total_delta_safe(spent_budget, slack)

        if slack == 0:
            return total_epsilon_naive, total_delta

        total_epsilon_drv = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum * np.log(1 / slack))
        total_epsilon_kov = epsilon_exp_sum + np.sqrt(2 * epsilon_sq_sum *
                                                      np.log(np.exp(1) + np.sqrt(epsilon_sq_sum) / slack))

        return min(total_epsilon_naive, total_epsilon_drv, total_epsilon_kov), total_delta

    def check_spend(self, epsilon, delta):
        """Checks if the provided budget can be spent while staying within the accountant's target budget.

        Parameters
        ----------
        epsilon : float
            Epsilon budget spend to check.

        delta : float
            Delta budget spend to check.

        Returns
        -------
        bool
            True if the budget can be spent, otherwise a :class:`.BudgetError` is raised..

        Raises
        ------
        BudgetError
            If the specified budget spend will result in the target budget being exceeded..

        """
        check_epsilon_delta(epsilon, delta)
        spent_budget = self.__spent_budget + [(epsilon, delta)]

        epsilon_spent, delta_spent = self.total_spent(spent_budget=spent_budget)

        if self.epsilon >= epsilon_spent and self.delta >= delta_spent:
            return True

        raise BudgetError("Privacy spend of ({},{}) not permissible; will exceed remaining privacy budget. "
                          "Use {}.{} to check remaining budget.".format(epsilon, delta, self.__class__.__name__,
                                                                        self.remaining_budget.__name__))

    def remaining_budget(self, k=1):
        """Calculates the budget that remains to be spent.

        Calculates the privacy budget that can be spent on `k` queries. Spending this budget on `k` queries will
        match the total budget, assuming no floating point errors.

        Parameters
        ----------
        k : int, default: 1
            The number of queries for which to calculate the remaining budget.

        Returns
        -------
        (epsilon, delta)
            Budget remaining to be spent on `k` queries.

        """
        if not isinstance(k, Integral):
            raise TypeError("k must be integer-valued, got {}.".format(type(k)))
        if k < 1:
            raise ValueError("k must be at least 1, got {}.".format(k))

        _, spent_delta = self.total_spent()
        delta = 1 - ((1 - self.delta) / (1 - spent_delta)) ** (1 / k) if spent_delta < 1.0 else 1.0
        # delta = 1 - np.exp((np.log(1 - self.delta) - np.log(1 - spent_delta)) / k)

        lower = 0
        upper = self.epsilon
        old_interval_size = (upper - lower) * 2

        while old_interval_size > upper - lower:
            old_interval_size = upper - lower
            mid = (upper + lower) / 2

            spent_budget = self.__spent_budget + [(mid, 0)] * k
            x_0, _ = self.total_spent(spent_budget=spent_budget)

            if x_0 >= self.epsilon:
                upper = mid
            if x_0 <= self.epsilon:
                lower = mid

        epsilon = (upper + lower) / 2

        return epsilon, delta

    def spend(self, epsilon, delta):
        """Spend the given privacy budget.

        Instructs the accountant to spend the given epsilon and delta privacy budget, while ensuring the target budget
        is not exceeded.

        Parameters
        ----------
        epsilon : float
            Epsilon privacy budget to spend.

        delta : float
            Delta privacy budget to spend.

        """
        check_epsilon_delta(epsilon, delta)

        self.check_spend(epsilon, delta)

        self.__spent_budget.append((epsilon, delta))

    @staticmethod
    def __total_delta_safe(spent_budget, slack):
        """
        Calculate total delta spend of `spent_budget`, with special consideration for floating point arithmetic.
        Should yield greater precision, especially for a large number of budget spends with very small delta.

        Parameters
        ----------
        spent_budget: list of tuples of the form (epsilon, delta)
            List of budget spends, for which the total delta spend is to be calculated.
        slack: float
            Delta slack parameter for composition of spends.

        Returns
        -------
        float
            Total delta spend.
        """
        delta_spend = [slack]
        for _, delta in spent_budget:
            delta_spend.append(delta)
        delta_spend.sort()

        # (1 - a) * (1 - b) = 1 - (a + b - a * b)
        prod = 0
        for delta in delta_spend:
            prod += delta - prod * delta

        return prod

"""
Implementation of the standard exponential mechanism, and its derivative, the hierarchical mechanism.
"""
from numbers import Real

import numpy as np
from numpy.random import random

from . import DPMechanism


class Exponential(DPMechanism):
    """
    The exponential mechanism, as first proposed by McSherry and Talwar.
    Paper link: https://www.cs.drexel.edu/~greenie/privacy/mdviadp.pdf
    """
    def __init__(self):
        super().__init__()
        self._domain_values = None
        self._utility_values = None
        self._normalising_constant = None
        self._sensitivity = None
        self._balanced_tree = False

    def __repr__(self):
        output = super().__repr__()
        output += ".set_utility(" + str(self.get_utility_list()) + ")" if self._utility_values is not None else ""

        return output

    def set_utility(self, utility_list):
        """
        Set the utility of the mechanism. Utilities define the pairwise distance between two entries of the mechanism's
        dictionary.

        :param utility_list: List of tuples, or list of lists, of the form ("label1", "label2", utility). Labels must
        be specified as strings (for non-string labels, a :class:`.DPTransformer` can be used), and the utility value
        must be a strictly positive `float`.
        :type utility_list: `list`
        :return: self.
        :rtype: :class:`.Exponential`
        """
        if not isinstance(utility_list, list):
            raise ValueError("Utility must be given in a list")

        self._normalising_constant = None

        utility_values = {}
        domain_values = []
        sensitivity = 0

        for _utility_sub_list in utility_list:
            value1, value2, utility_value = _utility_sub_list

            if not isinstance(value1, str) or not isinstance(value2, str):
                raise TypeError("Utility keys must be strings")
            if (value1.find("::") >= 0) or (value2.find("::") >= 0) \
                    or value1.endswith(":") or value2.endswith(":"):
                raise ValueError("Values cannot contain the substring \"::\" and cannot end in \":\". "
                                 "Use a DPTransformer if necessary.")
            if not isinstance(utility_value, Real):
                raise TypeError("Utility value must be a number")
            if utility_value < 0.0:
                raise ValueError("Utility values must be non-negative")

            sensitivity = max(sensitivity, utility_value)
            if value1 not in domain_values:
                domain_values.append(value1)
            if value2 not in domain_values:
                domain_values.append(value2)

            if value1 is value2:
                continue
            if value1 < value2:
                utility_values[value1 + "::" + value2] = utility_value
            else:
                utility_values[value2 + "::" + value1] = utility_value

        self._utility_values = utility_values
        self._sensitivity = sensitivity
        self._domain_values = domain_values

        self._check_utility_full(domain_values)

        return self

    def _check_utility_full(self, domain_values):
        for val1 in domain_values:
            for val2 in domain_values:
                if val1 >= val2:
                    continue

                if val1 + "::" + val2 not in self._utility_values:
                    raise ValueError("Utility value for %s missing" % (val1 + "::" + val2))

        return True

    def get_utility_list(self):
        """
        Get the list of utility values of the mechanism. Returned in the same format as accepted by
        :func:`.set_utility`.

        :return: Utility list of tuples, of the form ("label1", "label2", utility).
        :rtype: `list`
        """
        if self._utility_values is None:
            return None

        utility_list = []

        for _key, _value in self._utility_values.items():
            value1, value2 = _key.split("::", maxsplit=1)
            utility_list.append((value1, value2, _value))

        return utility_list

    def _build_normalising_constant(self, re_eval=False):
        balanced_tree = True
        first_constant_value = None
        normalising_constant = {}

        for _base_leaf in self._domain_values:
            constant_value = 0.0

            for _target_leaf in self._domain_values:
                constant_value += self._get_prob(_base_leaf, _target_leaf)

            normalising_constant[_base_leaf] = constant_value

            if first_constant_value is None:
                first_constant_value = constant_value
            elif not np.isclose(constant_value, first_constant_value):
                balanced_tree = False

        # If the tree is balanced, we can eliminate the doubling factor
        if balanced_tree and not re_eval:
            self._balanced_tree = True
            return self._build_normalising_constant(True)

        return normalising_constant

    def _get_utility(self, value1, value2):
        if value1 == value2:
            return 0

        if value1 > value2:
            return self._get_utility(value2, value1)

        return self._utility_values[value1 + "::" + value2]

    def _get_prob(self, value1, value2):
        if value1 == value2:
            return 1.0

        balancing_factor = 1 if self._balanced_tree else 2
        return np.exp(- self._epsilon * self._get_utility(value1, value2) / balancing_factor / self._sensitivity)

    def check_inputs(self, value):
        """
        Check that all parameters of the mechanism have been initialised correctly, and that the mechanism is ready
        to be used.

        :param value: Value to be checked.
        :type value: `string`
        :return: True if the mechanism is ready to be used.
        :rtype: `bool`
        """
        super().check_inputs(value)

        if self._utility_values is None:
            raise ValueError("Utility function must be set")

        if self._normalising_constant is None:
            self._normalising_constant = self._build_normalising_constant()

        if not isinstance(value, str):
            raise TypeError("Value to be randomised must be a string")

        if value not in self._domain_values:
            raise ValueError("Value \"%s\" not in domain" % value)

        return True

    def set_epsilon_delta(self, epsilon, delta):
        """
        Set the privacy parameters epsilon and delta for the mechanism.

        For the exponential mechanism, delta must be strictly zero.
        As is normal, epsilon must be strictly positive, epsilon >= 0.

        :param epsilon: Epsilon value of the mechanism.
        :type epsilon: `float`
        :param delta: Delta value of the mechanism. Must be zero for the exponential mechanism.
        :type delta: `float`
        :return: self
        :rtype: :class:`.Exponential`
        """
        if not delta == 0:
            raise ValueError("Delta must be zero")

        self._normalising_constant = None

        return super().set_epsilon_delta(epsilon, delta)

    def randomise(self, value):
        """
        Randomise the given value using the mechanism. The value must be an element of the mechanism dictionary.

        :param value: Value to be randomised.
        :type value: `string`
        :return: Randomised value.
        :rtype: `string`
        """
        self.check_inputs(value)

        unif_rv = random() * self._normalising_constant[value]
        cum_prob = 0

        for _target_value in self._normalising_constant.keys():
            cum_prob += self._get_prob(value, _target_value)

            if unif_rv <= cum_prob:
                return _target_value

        return None


class ExponentialHierarchical(Exponential):
    """
    Adaptation of the exponential mechanism to hierarchical data. Simplifies the process of specifying utility values,
    as the values can be inferred from the hierarchy.
    """
    def __init__(self):
        super().__init__()
        self._list_hierarchy = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_hierarchy(" + str(self._list_hierarchy) + ")" if self._list_hierarchy is not None else ""

        return output

    def _build_hierarchy(self, nested_list, parent_node=None):
        if parent_node is None:
            parent_node = []

        hierarchy = {}

        for _i, _value in enumerate(nested_list):
            if isinstance(_value, str):
                hierarchy[_value] = parent_node + [_i]
            elif not isinstance(_value, list):
                raise TypeError("All leaves of the hierarchy must be a string " +
                                "(see node " + (parent_node + [_i]).__str__() + ")")
            else:
                hierarchy.update(self._build_hierarchy(_value, parent_node + [_i]))

        self._check_hierarchy_height(hierarchy)

        return hierarchy

    @staticmethod
    def _check_hierarchy_height(hierarchy):
        hierarchy_height = None
        for _value, _hierarchy_locator in hierarchy.items():
            if hierarchy_height is None:
                hierarchy_height = len(_hierarchy_locator)
            elif len(_hierarchy_locator) != hierarchy_height:
                raise ValueError("Leaves of the hierarchy must all be at the same level " +
                                 "(node %s is at level %d instead of hierarchy height %d)" %
                                 (_hierarchy_locator.__str__(), len(_hierarchy_locator), hierarchy_height))
        return None

    @staticmethod
    def _build_utility_list(hierarchy):
        if not isinstance(hierarchy, dict):
            raise TypeError("Hierarchy for _build_utility_list must be a dict")

        utility_list = []
        hierarchy_height = None

        for _root_value, _root_hierarchy_locator in hierarchy.items():
            if hierarchy_height is None:
                hierarchy_height = len(_root_hierarchy_locator)

            for _target_value, _target_hierarchy_locator in hierarchy.items():
                if _root_value >= _target_value:
                    continue

                i = 0
                while (i < len(_root_hierarchy_locator) and
                       _root_hierarchy_locator[i] == _target_hierarchy_locator[i]):
                    i += 1

                utility_list.append([_root_value, _target_value, hierarchy_height - i])

        return utility_list

    def set_hierarchy(self, list_hierarchy):
        """
        Set the hierarchy of the mechanism, specified as a list of lists. The hierarchy must have a uniform height, with
        values specified as strings. For non-string values, a :class:`.DPTransformer` can be used. The exponential
        mechanism is then invoked, using the height of the closest ancestor as the utility metric.

        :param list_hierarchy: Hierarchy list.
        :type list_hierarchy: `list`
        :return: self.
        :rtype: :class:`.ExponentialHierarchical`
        """
        if not isinstance(list_hierarchy, list):
            raise TypeError("Hierarchy must be a list")

        self._list_hierarchy = list_hierarchy
        hierarchy = self._build_hierarchy(list_hierarchy)
        self.set_utility(self._build_utility_list(hierarchy))

        return self

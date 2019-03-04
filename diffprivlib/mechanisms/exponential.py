from numbers import Real

from numpy import exp
from numpy.random import random

from . import DPMechanism


class Exponential(DPMechanism):
    def __init__(self):
        super().__init__()
        self._utility_function = None
        self._normalising_constant = None
        self._sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".set_utility(" + str(self.get_utility_list()) + ")" if self._utility_function is not None else ""

        return output

    def set_utility(self, utility_list):
        if self._epsilon is None:
            raise RuntimeError("Epsilon must be set before utility is set")

        if utility_list is None:
            return self

        if type(utility_list) is not list:
            raise ValueError("Utility must be in the form of a list")

        utility_function = {}
        domain_values = []
        sensitivity = 0

        for _utility_sub_list in utility_list:
            value1 = _utility_sub_list[0]
            value2 = _utility_sub_list[1]
            utility_value = _utility_sub_list[2]

            if (type(value1) is not str) or (type(value2) is not str):
                raise TypeError("Utility keys must be strings")
            if (value1.find("::") >= 0) or (value2.find("::") >= 0) \
                    or value1.endswith(":") or value2.endswith(":"):
                raise ValueError("Values cannot contain the substring \"::\""
                                 " and cannot end in \":\". Use a DPTransformer if necessary.")
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
                utility_function[value1 + "::" + value2] = utility_value
            else:
                utility_function[value2 + "::" + value1] = utility_value

        self._utility_function = utility_function
        self._sensitivity = sensitivity

        self._check_utility_full(domain_values)

        self._normalising_constant = self._build_normalising_constant(domain_values)

        return self

    def _check_utility_full(self, domain_values):
        for val1 in domain_values:
            for val2 in domain_values:
                if val1 >= val2:
                    continue

                if val1 + "::" + val2 not in self._utility_function:
                    raise ValueError("Utility value for %s missing" % (val1 + "::" + val2))

    def get_utility_list(self):
        if self._utility_function is None:
            return None

        utility_list = []

        for _key, _value in self._utility_function.items():
            value1, value2 = _key.split("::", maxsplit=1)
            utility_list.append([value1, value2, _value])

        return utility_list

    def _build_normalising_constant(self, domain_values, re_eval=False):
        balanced_hierarchy = True
        first_constant_value = None
        normalising_constant = {}

        for _baseLeaf in domain_values:
            constant_value = 0.0

            for _targetLeaf in domain_values:
                constant_value += self.get_prob(_baseLeaf, _targetLeaf)

            normalising_constant[_baseLeaf] = constant_value

            if first_constant_value is None:
                first_constant_value = constant_value
            elif constant_value != first_constant_value:
                balanced_hierarchy = False

        if balanced_hierarchy and not re_eval:
            self._sensitivity /= 2
            return self._build_normalising_constant(domain_values, True)

        return normalising_constant

    def _get_utility(self, value1, value2):
        if value1 == value2:
            return 0

        if value1 > value2:
            return self._get_utility(value2, value1)

        return self._utility_function[value1 + "::" + value2]

    def get_prob(self, value1, value2):
        return exp(- self._epsilon * self._get_utility(value1, value2) / self._sensitivity)

    def check_inputs(self, value):
        super().check_inputs(value)

        if self._utility_function is None:
            raise ValueError("Utility function must be set")

        if type(value) is not str:
            raise TypeError("Value to be randomised must be a string")

        if value not in self._normalising_constant:
            raise ValueError("Value \"%s\" not in domain" % value)

        return True

    def set_epsilon_delta(self, epsilon, delta):
        if delta > 0:
            raise ValueError("Delta must be zero")

        return super().set_epsilon_delta(epsilon, delta)

    def randomise(self, value):
        self.check_inputs(value)

        u = random() * self._normalising_constant[value]
        cum_prob = 0

        for _targetValue in self._normalising_constant.keys():
            cum_prob += self.get_prob(value, _targetValue)

            if u <= cum_prob:
                return _targetValue

        return None


class ExponentialHierarchical(Exponential):
    def __init__(self):
        super().__init__()
        self._list_hierarchy = None

    def __repr__(self):
        output = super(Exponential, self).__repr__()
        output += ".set_hierarchy(" + str(self._list_hierarchy) + ")" if self._list_hierarchy is not None else ""

        return output

    def _build_hierarchy(self, nested_list, parent_node=None):
        if parent_node is None:
            parent_node = []

        hierarchy = {}

        for _i, _value in enumerate(nested_list):
            if type(_value) is str:
                hierarchy[_value] = parent_node + [_i]
            elif type(_value) is not list:
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
        if type(hierarchy) is not dict:
            raise TypeError("Hierarchy must be of type dict")

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
        if self._epsilon is None:
            raise RuntimeError("Epsilon must be set before hierarchy is set")

        if list_hierarchy is None:
            return self

        if type(list_hierarchy) is not list:
            raise ValueError("Hierarchy must be a list")

        self._list_hierarchy = list_hierarchy
        hierarchy = self._build_hierarchy(list_hierarchy)
        self.set_utility(self._build_utility_list(hierarchy))

        return self

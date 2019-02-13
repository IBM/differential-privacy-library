from abc import ABC, abstractmethod
import numpy as np
from random import random
from numbers import Number
from copy import copy, deepcopy


class DPMachine(ABC):
    @abstractmethod
    def randomise(self, value):
        pass

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)

    @abstractmethod
    def set_epsilon(self, epsilon):
        pass


class DPMechanism(DPMachine, ABC):
    def __init__(self):
        self.epsilon = None
        self.delta = None

    def __repr__(self):
        output = str(self.__module__) + "." + str(self.__class__.__name__) + "()"
        output += ".setEpsilon(" + str(self.epsilon) + ")" if self.epsilon is not None else ""

        return output

    @abstractmethod
    def randomise(self, value):
        pass

    def get_bias(self, value):
        return None

    def get_variance(self, value):
        return None
    
    def get_mse(self, value):
        return self.get_variance(value) + (self.get_bias(value)) ** 2 if self.get_variance(value) is not None else None

    def set_epsilon(self, epsilon):
        if self.epsilon is not None:
            raise ValueError("Epsilon cannot be reset; initiate a new mechanism instance instead.")

        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive")

        self.epsilon = epsilon
        return self

    def set_epsilon_delta(self, epsilon, delta):
        self.set_epsilon(epsilon)

        if 0 <= delta <= 1:
            self.delta = delta
        else:
            raise ValueError("Delta must be in [0, 1]")

        return self

    def check_inputs(self, value):
        if self.epsilon is None:
            raise ValueError("Epsilon must be set")
        return True


class TruncationMachine:
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None

    def __repr__(self):
        output = ".setBounds(" + str(self.lower_bound) + ", " + str(self.upper_bound) + ")"\
            if self.lower_bound is not None else ""
        
        return output

    def set_bounds(self, lower, upper):
        if (not isinstance(lower, Number)) or (not isinstance(upper, Number)):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")
        
        self.lower_bound = lower
        self.upper_bound = upper
        
        return self
        
    def check_inputs(self, value):
        if (self.lower_bound is None) or (self.upper_bound is None):
            raise ValueError("Upper and lower bounds must be set")
        return True
          
    def truncate(self, value):
        if value > self.upper_bound:
            return self.upper_bound
        elif value < self.lower_bound:
            return self.lower_bound

        return value


class FoldingMachine:
    def __init__(self):
        self.lower_bound = None
        self.upper_bound = None
        
    def __repr__(self):
        output = ".setBounds(" + str(self.lower_bound) + ", " + str(self.upper_bound) + ")" \
            if self.lower_bound is not None else ""
        
        return output
        
    def set_bounds(self, lower, upper):
        if (not isinstance(lower, Number)) or (not isinstance(upper, Number)):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")
        
        self.lower_bound = lower
        self.upper_bound = upper
        
        return self
        
    def check_inputs(self, value):
        if (self.lower_bound is None) or (self.upper_bound is None):
            raise ValueError("Upper and lower bounds must be set")
        return True

    def fold(self, value):
        if value < self.lower_bound:
            return self.fold(2 * self.lower_bound - value)
        if value > self.upper_bound:
            return self.fold(2 * self.upper_bound - value)

        return value


class LaplaceMechanism(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setSensitivity(" + str(self.sensitivity) + ")" if self.sensitivity is not None else ""
        
        return output

    def set_sensitivity(self, sensitivity):
        """

        :param sensitivity: The sensitivity of the function being considered
        :type sensitivity: `float`
        :return:
        """

        if not isinstance(sensitivity, Number):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self.sensitivity = sensitivity
        return self
        
    def check_inputs(self, value):
        super().check_inputs(value)
        
        if not isinstance(value, Number):
            raise TypeError("Value to be randomised must be a number")

        if self.sensitivity is None:
            raise ValueError("Sensitivity must be set")

        return True

    def get_bias(self, value):
        return 0.0

    def get_variance(self, value):
        self.check_inputs(0)

        return 2 * (self.sensitivity / self.epsilon) ** 2

    def randomise(self, value):
        self.check_inputs(value)
        
        shape = self.sensitivity / self.epsilon
        
        u = random() - 0.5

        return value - shape * np.sign(u) * np.log(1 - 2 * np.abs(u))


class TruncatedLaplaceMechanism(LaplaceMechanism, TruncationMachine):
    def __init__(self):
        super().__init__()
        TruncationMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationMachine.__repr__(self)
        
        return output

    def get_bias(self, value):
        self.check_inputs(value)

        shape = self.sensitivity / self.epsilon

        return shape / 2 * (np.exp((self.lower_bound - value) / shape) - np.exp((value - self.upper_bound) / shape))

    def get_variance(self, value):
        self.check_inputs(value)

        shape = self.sensitivity / self.epsilon

        variance = value ** 2 + shape * (self.lower_bound * np.exp((self.lower_bound - value) / shape)
                                         - self.upper_bound * np.exp((value - self.upper_bound) / shape))
        variance += (shape ** 2) * (2 - np.exp((self.lower_bound - value) / shape)
                                    - np.exp((value - self.upper_bound) / shape))

        variance -= (self.get_bias(value) + value) ** 2

        return variance

    def check_inputs(self, value):
        super().check_inputs(value)
        TruncationMachine.check_inputs(self, value)

        return True

    def randomise(self, value):
        TruncationMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return super().truncate(noisy_value)


class FoldedLaplaceMechanism(LaplaceMechanism, FoldingMachine):
    def __init__(self):
        super().__init__()
        FoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += FoldingMachine.__repr__(self)

        return output

    def get_bias(self, value):
        self.check_inputs(value)

        shape = self.sensitivity / self.epsilon

        bias = shape * (np.exp((self.lower_bound + self.upper_bound - 2 * value) / shape) - 1)
        bias /= np.exp((self.lower_bound - value) / shape) + np.exp((self.upper_bound - value) / shape)

        return bias

    def check_inputs(self, value):
        super().check_inputs(value)
        FoldingMachine.check_inputs(self, value)

        return True

    def randomise(self, value):
        FoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return super().fold(noisy_value)


class BoundedLaplaceMechanism(TruncatedLaplaceMechanism):
    def __init__(self):
        super().__init__()
        self.shape = None

    def __find_shape(self):
        eps = self.epsilon
        delta = 0.0
        diam = self.upper_bound - self.lower_bound
        dq = self.sensitivity

        def delta_c(shape):
            return (2 - np.exp(- dq / shape) - np.exp(- (diam - dq) / shape)) / (1 - np.exp(- diam / shape))

        def f(shape):
            return dq / (eps - np.log(delta_c(shape)) - np.log(1 - delta))

        left = dq / (eps - np.log(1 - delta))
        right = f(left)
        old_interval_size = (right - left) * 2

        while old_interval_size > right - left:
            old_interval_size = right - left
            middle = (right + left)/2

            if f(middle) >= middle:
                left = middle
            if f(middle) <= middle:
                right = middle

        return (right + left) / 2

    @staticmethod
    def __cdf(x, shape):
        if x < 0:
            return 0.5 * np.exp(x / shape)
        else:
            return 1 - 0.5 * np.exp(-x / shape)

    def get_effective_epsilon(self):
        if self.shape is None:
            self.shape = self.__find_shape()

        return self.sensitivity / self.shape

    def get_bias(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = self.__find_shape()

        bias = (self.shape - self.lower_bound + value) / 2 * np.exp((self.lower_bound - value) / self.shape) \
            - (self.shape + self.upper_bound - value) / 2 * np.exp((value - self.upper_bound) / self.shape)
        bias /= 1 - np.exp((self.lower_bound - value) / self.shape) / 2 \
            - np.exp((value - self.upper_bound) / self.shape) / 2

        return bias

    def get_variance(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = self.__find_shape()

        variance = value**2
        variance -= (np.exp((self.lower_bound - value) / self.shape) * (self.lower_bound ** 2)
                     + np.exp((value - self.upper_bound) / self.shape) * (self.upper_bound ** 2)) / 2
        variance += self.shape * (self.lower_bound * np.exp((self.lower_bound - value) / self.shape)
                                  - self.upper_bound * np.exp((value - self.upper_bound) / self.shape))
        variance += (self.shape ** 2) * (2 - np.exp((self.lower_bound - value) / self.shape)
                                         - np.exp((value - self.upper_bound) / self.shape))
        variance /= 1 - (np.exp(-(value - self.lower_bound) / self.shape)
                         + np.exp(-(self.upper_bound - value) / self.shape)) / 2

        variance -= (self.get_bias(value) + value) ** 2

        return variance

    def randomise(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = self.__find_shape()

        value = min(value, self.upper_bound)
        value = max(value, self.lower_bound)

        u = random()
        u *= self.__cdf(self.upper_bound - value, self.shape) - self.__cdf(self.lower_bound - value, self.shape)
        u += self.__cdf(self.lower_bound - value, self.shape)
        u -= 0.5

        return value - self.shape * np.sign(u) * np.log(1 - 2 * np.abs(u))


class GeometricMechanism(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None
        self.shape = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setSensitivity(" + str(self.sensitivity) + ")" if self.sensitivity is not None else ""

        return output

    def set_sensitivity(self, sensitivity):
        """

        :param sensitivity:
        :type sensitivity `float`
        :return:
        """
        if not isinstance(sensitivity, Number):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self.sensitivity = sensitivity
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if not isinstance(value, Number):
            raise TypeError("Value to be randomised must be a number")

        if self.sensitivity is None:
            raise ValueError("Sensitivity must be set")

    def randomise(self, value):
        self.check_inputs(value)

        if self.shape is None:
            self.shape = - self.epsilon / self.sensitivity

        # Need to account for overlap of 0-value between distributions of different sign
        u = random() - 0.5
        u *= 1 + np.exp(self.shape)
        sgn = -1 if u < 0 else 1

        # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
        return int(value + sgn * np.floor(np.log(sgn * u) / self.shape))

    def old_randomise(self, value):
        self.check_inputs(value)

        shape = self.epsilon / self.sensitivity

        u = random() - 0.5
        sgn = np.sign(u)
        u *= sgn * (np.exp(shape) + 1)/(np.exp(shape) - 1)

        cum_prob = -0.5
        i = -1
        while u > cum_prob:
            i += 1
            cum_prob += np.exp(-shape * i)

        return int(value + sgn * i)


class TruncatedGeometricMechanism(GeometricMechanism, TruncationMachine):
    def __init__(self):
        super().__init__()
        TruncationMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += TruncationMachine.__repr__(self)

        return output

    def randomise(self, value):
        TruncationMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return int(super().truncate(noisy_value))


class FoldedGeometricMechanism(GeometricMechanism, FoldingMachine):
    def __init__(self):
        super().__init__()
        FoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += FoldingMachine.__repr__(self)

        return output

    def randomise(self, value):
        FoldingMachine.check_inputs(self, value)

        noisy_value = super().randomise(value)
        return super().fold(noisy_value)


class ExponentialMechanism(DPMechanism):
    def __init__(self):
        super().__init__()
        self.utility_function = None
        self.normalising_constant = None
        self.sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setUtility(" + str(self.get_utility_list()) + ")" if self.utility_function is not None else ""

        return output

    def set_utility(self, utility_list):
        if self.epsilon is None:
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
            utility_value = float(_utility_sub_list[2])

            if (type(value1) is not str) or (type(value2) is not str):
                raise ValueError("Utility keys must be strings")
            if (value1.find("::") >= 0) or (value2.find("::") >= 0) \
                    or value1.endswith(":") or value2.endswith(":"):
                raise ValueError("Values cannot contain the substring \"::\""
                                 " and cannot end in \":\". Use a DPTransformer if necessary.")
            if not isinstance(utility_value, Number):
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

        self.utility_function = utility_function
        self.sensitivity = sensitivity
        self.normalising_constant = self.__build_normalising_constant(domain_values)

        return self

    def get_utility_list(self):
        if self.utility_function is None:
            return None

        utility_list = []

        for _key, _value in self.utility_function.items():
            value1, value2 = _key.split("::", maxsplit=1)
            utility_list.append([value1, value2, _value])

        return utility_list

    def __build_normalising_constant(self, domain_values, re_eval=False):
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
            self.sensitivity /= 2
            return self.__build_normalising_constant(domain_values, True)

        return normalising_constant

    def __get_utility(self, value1, value2):
        if value1 == value2:
            return 0

        if value1 > value2:
            return self.__get_utility(value2, value1)

        return self.utility_function[value1 + "::" + value2]

    def get_prob(self, value1, value2):
        return np.exp(- self.epsilon * self.__get_utility(value1, value2) / self.sensitivity)

    def check_inputs(self, value):
        super().check_inputs(value)

        if self.utility_function is None:
            raise ValueError("Utility function must be set")

        if type(value) is not str:
            raise TypeError("Value to be randomised must be a string")

        if value not in self.normalising_constant:
            raise ValueError("Value \"%s\" not in domain" % value)

    def randomise(self, value):
        self.check_inputs(value)

        u = random() * self.normalising_constant[value]
        cum_prob = 0

        for _targetValue in self.normalising_constant.keys():
            cum_prob += self.get_prob(value, _targetValue)

            if u <= cum_prob:
                return _targetValue

        return None


class HierarchicalMechanism(ExponentialMechanism):
    def __init__(self):
        super().__init__()
        self.list_hierarchy = None

    def __repr__(self):
        output = super(ExponentialMechanism, self).__repr__()
        output += ".setHierarchy(" + str(self.list_hierarchy) + ")" if self.list_hierarchy is not None else ""

        return output

    def __build_hierarchy(self, nested_list, parent_node=None):
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
                hierarchy.update(self.__build_hierarchy(_value, parent_node + [_i]))

        self.__check_hierarchy_height(hierarchy)

        return hierarchy

    @staticmethod
    def __check_hierarchy_height(hierarchy):
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
    def __build_utility_list(hierarchy):
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
        if self.epsilon is None:
            raise RuntimeError("Epsilon must be set before hierarchy is set")

        if list_hierarchy is None:
            return self

        if type(list_hierarchy) is not list:
            raise ValueError("Hierarchy must be a list")

        self.list_hierarchy = list_hierarchy
        hierarchy = self.__build_hierarchy(list_hierarchy)
        self.set_utility(self.__build_utility_list(hierarchy))

        return self


class BinaryMechanism(DPMechanism):
    def __init__(self):
        super().__init__()
        self.value1 = None
        self.value2 = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setLabels(" + str(self.value1) + ", " + str(self.value2) + ")" if self.value1 is not None else ""

        return output

    def set_labels(self, value1, value2):
        if (type(value1) is not str) or (type(value2) is not str):
            raise ValueError("Binary labels must be strings")

        if len(value1) * len(value2) == 0:
            raise ValueError("Binary labels must be non-empty strings")

        if value1 == value2:
            raise ValueError("Binary labels must be different")

        self.value1 = value1
        self.value2 = value2
        return self

    def check_inputs(self, value):
        super().check_inputs(value)

        if (self.value1 is None) or (self.value2 is None):
            raise ValueError("Binary labels must be set")

        if type(value) is not str:
            raise ValueError("Value to be randomised must be a string")

        if value not in [self.value1, self.value2]:
            raise ValueError("Value to be randomised is not in the domain")

    def randomise(self, value):
        self.check_inputs(value)

        indicator = 0 if value == self.value1 else 1

        u = random() * (np.exp(self.epsilon) + 1)

        if u > np.exp(self.epsilon):
            indicator = 1 - indicator

        return self.value1 if indicator == 0 else self.value2


class DPTransformer(DPMachine):
    def __init__(self, parent):
        if not isinstance(parent, DPMachine):
            raise TypeError("Data transformer must take a DPMachine as input")

        self.parent = parent

    def pre_transform(self, value):
        return value

    def post_transform(self, value):
        return value

    def set_epsilon(self, epsilon):
        self.parent.set_epsilon(epsilon)
        return self

    def randomise(self, value):
        transformed_value = self.pre_transform(value)
        noisy_value = self.parent.randomise(transformed_value)
        output_value = self.post_transform(noisy_value)
        return output_value


class RoundedInteger(DPTransformer):
    def post_transform(self, value):
        return int(np.round(value))


class StringToInt(DPTransformer):
    def pre_transform(self, value):
        return int(value)

    def post_transform(self, value):
        return str(value)


class IntToString(DPTransformer):
    def pre_transform(self, value):
        return str(value)

    def post_transform(self, value):
        return int(value)

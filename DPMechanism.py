import numpy as np
from random import random
from abc import ABC, abstractmethod
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

class DPMechanism(DPMachine, ABC):
    def __init__(self):
        self.epsilon = None

    def __repr__(self):
        output = str(self.__module__) + "." + str(self.__class__.__name__) + "()"
        output += ".setEpsilon(" + str(self.epsilon) + ")" if self.epsilon is not None else ""

        return output

    @abstractmethod
    def randomise(self, value):
        pass

    def getBias(self, value):
        return None

    def setEpsilon(self, epsilon):
        if self.epsilon is not None:
            raise ValueError("Epsilon cannot be reset; initiate a new mechanism instance instead.")

        if epsilon <= 0:
            raise ValueError("Epsilon must be strictly positive")

        self.epsilon = epsilon
        return self

    def checkInputs(self, value):
        if self.epsilon is None:
            raise ValueError("Epsilon must be set")
        return True

class TruncationMachine():
    def __init__(self):
        self.lowerBound = None
        self.upperBound = None

    def __repr__(self):
        output = ".setBounds(" + str(self.lowerBound) + ", " + str(self.upperBound) + ")" if self.lowerBound is not None else ""
        
        return output
        
    def setBounds(self, lower, upper):
        if (not isinstance(lower, Number)) or (not isinstance(upper, Number)):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")
        
        self.lowerBound = lower
        self.upperBound = upper
        
        return self
        
    def checkInputs(self, value):
        if (self.lowerBound is None) or (self.upperBound is None):
            raise ValueError("Upper and lower bounds must be set")
        return True
          
    def truncate(self, value):
        if value > self.upperBound:
            return self.upperBound
        elif value < self.lowerBound:
            return self.lowerBound

        return value

class FoldingMachine():
    def __init__(self):
        self.lowerBound = None
        self.upperBound = None
        
    def __repr__(self):
        output = ".setBounds(" + str(self.lowerBound) + ", " + str(self.upperBound) + ")" if self.lowerBound is not None else ""
        
        return output
        
    def setBounds(self, lower, upper):
        if (not isinstance(lower, Number)) or (not isinstance(upper, Number)):
            raise TypeError("Bounds must be numeric")

        if lower > upper:
            raise ValueError("Lower bound must not be greater than upper bound")
        
        self.lowerBound = lower
        self.upperBound = upper
        
        return self
        
    def checkInputs(self, value):
        if (self.lowerBound is None) or (self.upperBound is None):
            raise ValueError("Upper and lower bounds must be set")
        return True

    def fold(self, value):
        if value < self.lowerBound:
            return self.fold(2 * self.lowerBound - value)
        if value > self.upperBound:
            return self.fold(2 * self.upperBound - value)

        return value

class LaplaceMechanism(DPMechanism):
    def __init__(self):
        super().__init__()
        self.sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setSensitivity(" + str(self.sensitivity) + ")" if self.sensitivity is not None else ""
        
        return output

    def setSensitivity(self, sensitivity):
        if not isinstance(sensitivity, Number):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self.sensitivity = sensitivity
        return self
        
    def checkInputs(self, value):
        super().checkInputs(value)
        
        if not isinstance(value, Number):
            raise TypeError("Value to be randomised must be a number")

        if self.sensitivity is None:
            raise ValueError("Sensitivity must be set")

        return True

    def getBias(self, value):
        return 0.0

    def randomise(self, value):
        self.checkInputs(value)
        
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

    def getBias(self, value):
        self.checkInputs(value)
        TruncationMachine.checkInputs(self, value)

        shape = self.sensitivity / self.epsilon

        return shape / 2 * (np.exp((self.lowerBound - value) / shape) - np.exp((value - self.upperBound) / shape))

    def randomise(self, value):
        TruncationMachine.checkInputs(self, value)

        noisyValue = super().randomise(value)
        return super().truncate(noisyValue)

class FoldedLaplaceMechanism(LaplaceMechanism, FoldingMachine):
    def __init__(self):
        super().__init__()
        FoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += FoldingMachine.__repr__(self)
        
        return output

    def getBias(self, value):
        shape = self.sensitivity / self.epsilon
        # l = (self.lowerBound - value) / b
        # u = (self.upperBound - value) / b

        # bias = np.exp(l) * (1 - l) - np.exp(-u) *(1 + u)
        # bias += np.exp(2 * l) * (1 + np.exp(-2 * u)) * (np.exp(-l) * (l + 1) - np.exp(-u) * (u + 1)) / (1 - np.exp(2 * l - 2 * u))
        # bias += np.exp(-2 * u) * (1 + np.exp(2 * l)) * (np.exp(u) * (u - 1) - np.exp(l) * (l - 1)) / (1 - np.exp(2 * l - 2 * u))
        # bias *= b / 2

        bias = shape * (np.exp((self.lowerBound + self.upperBound - 2 * value) / shape) - 1)
        bias /= np.exp((self.lowerBound - value) / shape) + np.exp((self.upperBound - value) / shape)

        return bias

    def randomise(self, value):
        FoldingMachine.checkInputs(self, value)

        noisyValue = super().randomise(value)
        return super().fold(noisyValue)

class BoundedLaplaceMechanism(TruncatedLaplaceMechanism):
    def __init__(self):
        super().__init__()
        self.shape = None

    def __findShape(self):
        EPS = self.epsilon
        DEL = 0.0
        DIAM = self.upperBound - self.lowerBound
        DQ = self.sensitivity

        def deltaC(shape):
            return (2 - np.exp(- DQ / shape) - np.exp(- (DIAM - DQ) / shape)) / (1 - np.exp(- DIAM / shape))

        def f(shape):
            return DQ / (EPS - np.log(deltaC(shape)) - np.log(1 - DEL))

        left = DQ / (EPS - np.log(1 - DEL))
        right = f(left)
        oldIntervalSize = (right - left) * 2

        while (oldIntervalSize > right - left):
            oldIntervalSize = right - left
            middle = (right + left)/2

            if (f(middle) >= middle): left = middle
            if (f(middle) <= middle): right = middle

        return (right + left) / 2   

    def __cdf(self, x, shape):
        if (x < 0):
            return 0.5 * np.exp(x / shape)
        else:
            return 1 - 0.5 * np.exp(-x / shape)

    def getEffectiveEpsilon(self):
        if self.shape is None:
            self.shape = self.__findShape()

        return self.sensitivity / self.shape

    def getBias(self, value):
        self.checkInputs(value)

        if self.shape is None:
            self.shape = self.__findShape()

        bias = (self.shape - self.lowerBound + value) / 2 * np.exp((self.lowerBound - value) / self.shape)\
            - (self.shape + self.upperBound - value) / 2 * np.exp((value - self.upperBound) / self.shape)
        bias /= 1 - np.exp((self.lowerBound - value) / self.shape) / 2 - np.exp((value - self.upperBound) / self.shape) / 2

        return bias
        
    def randomise(self, value):
        self.checkInputs(value)
        
        if self.shape is None:
            self.shape = self.__findShape()
        
        value = min(value, self.upperBound)
        value = max(value, self.lowerBound)
        
        u = random()
        u *= self.__cdf(self.upperBound - value, self.shape) - self.__cdf(self.lowerBound - value, self.shape)
        u += self.__cdf(self.lowerBound - value, self.shape)
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
        
    def setSensitivity(self, sensitivity):
        if not isinstance(sensitivity, Number):
            raise TypeError("Sensitivity must be numeric")

        if sensitivity <= 0:
            raise ValueError("Sensitivity must be strictly positive")

        self.sensitivity = sensitivity
        return self

    def checkInputs(self, value):
        super().checkInputs(value)
        
        if not isinstance(value, Number):
            raise TypeError("Value to be randomised must be a number")

        if self.sensitivity is None:
            raise ValueError("Sensitivity must be set")

    def randomise(self, value):
        self.checkInputs(value)
        
        if self.shape is None:
            self.shape = - self.epsilon / self.sensitivity

        # Need to account for overlap of 0-value between distributions of different sign
        u = random() - 0.5 
        u *= 1 + np.exp(self.shape) 
        sgn = -1 if u < 0 else 1
        
        # Use formula for geometric distribution, with ratio of exp(-epsilon/sensitivity)
        return int(value + sgn * np.floor(np.log(sgn * u) / self.shape))

    def oldRandomise(self, value):
        self.checkInputs(value)

        shape = self.epsilon / self.sensitivity

        u = random() - 0.5
        sgn = np.sign(u)
        u *= sgn * (np.exp(shape) + 1)/(np.exp(shape) - 1)

        cumProb = -0.5
        i = -1
        while u > cumProb:
            i += 1
            cumProb += np.exp(-shape * i)

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
        TruncationMachine.checkInputs(self, value)

        noisyValue = super().randomise(value)
        return int(super().truncate(noisyValue))

class FoldedGeometricMechanism(GeometricMechanism, FoldingMachine):
    def __init__(self):
        super().__init__()
        FoldingMachine.__init__(self)

    def __repr__(self):
        output = super().__repr__()
        output += FoldingMachine.__repr__(self)
        
        return output

    def randomise(self, value):
        FoldingMachine.checkInputs(self, value)

        noisyValue = super().randomise(value)
        return super().fold(noisyValue)

class ExponentialMechanism(DPMechanism):
    def __init__(self):
        super().__init__()
        self.utilityFunction = None
        self.normalisingConstant = None
        self.sensitivity = None

    def __repr__(self):
        output = super().__repr__()
        output += ".setUtility(" + str(self.getUtilityList()) + ")" if self.utilityFunction is not None else ""
        
        return output

    def setUtility(self, utilityList):
        if (self.epsilon == None):
            raise RuntimeError("Epsilon must be set before utility is set")

        if utilityList == None:
            return self

        if type(utilityList) is not list:
            raise ValueError("Utility must be in the form of a list")

        utilityFunction = {}
        domainValues = []
        sensitivity = 0

        for _utilitySubList in utilityList:
            value1 = _utilitySubList[0]
            value2 = _utilitySubList[1]
            utilityValue = _utilitySubList[2]

            if (type(value1) is not str) or (type(value2) is not str):
                raise ValueError("Utility keys must be strings")
            if (value1.find("::") >= 0) or (value2.find("::") >= 0)\
                or value1.endswith(":") or value2.endswith(":"):
                raise ValueError("Values cannot contain the substring \"::\""
                    " and cannot end in \":\". Use a DPTransformer if necessary.")
            if not isinstance(utilityValue, Number):
                raise TypeError("Utility value must be a number")
            if utilityValue < 0:
                raise ValueError("Utility values must be non-negative")
            
            sensitivity = max(sensitivity, utilityValue)
            if value1 not in domainValues: domainValues.append(value1)
            if value2 not in domainValues: domainValues.append(value2)

            if (value1 is value2):
                continue
            if value1 < value2:
                utilityFunction[value1 + "::" + value2] = utilityValue
            else:
                utilityFunction[value2 + "::" + value1] = utilityValue

        self.utilityFunction = utilityFunction
        self.sensitivity = sensitivity
        self.normalisingConstant = self.__buildNormalisingConstant(domainValues)
        
        return self

    def getUtilityList(self):
        if self.utilityFunction == None:
            return None

        utilityList = []

        for _key, _value in self.utilityFunction.items():
            value1, value2 = _key.split("::", maxsplit=1)
            utilityList.append([value1, value2, _value])

        return utilityList
        
    def __buildNormalisingConstant(self, domainValues, reEval = False):
        balancedHierarchy = True
        firstConstantValue = None
        normalisingConstant = {}
        
        for _baseLeaf in domainValues:
            constantValue = 0.0
            
            for _targetLeaf in domainValues:
                constantValue += self.getProb(_baseLeaf, _targetLeaf)
        
            normalisingConstant[_baseLeaf] = constantValue
            
            if firstConstantValue == None:
                firstConstantValue = constantValue
            elif constantValue != firstConstantValue:
                balancedHierarchy = False
            
        if balancedHierarchy and not reEval:
            self.sensitivity /= 2
            return self.__buildNormalisingConstant(domainValues, True)
            
        return normalisingConstant

    def __getUtility(self, value1, value2):
        if value1 == value2:
            return 0
        
        if value1 > value2:
            return self.__getUtility(value2, value1)
        
        return self.utilityFunction[value1 + "::" + value2]
    
    def getProb(self, value1, value2):
        return np.exp(- self.epsilon * self.__getUtility(value1, value2) / self.sensitivity)
    
    def checkInputs(self, value):
        super().checkInputs(value)
        
        if self.utilityFunction == None:
            raise ValueError("Utility function must be set")
        
        if type(value) is not str:
            raise TypeError("Value to be randomised must be a string")

        if value not in self.normalisingConstant:
            raise ValueError("Value \"%s\" not in domain" % value)
            
    def randomise(self, value):
        self.checkInputs(value)
        
        u = random() * self.normalisingConstant[value]
        cumProb = 0
        
        for _targetValue in self.normalisingConstant.keys():
            cumProb += self.getProb(value, _targetValue)
            
            if u <= cumProb:
                return _targetValue
        
        return None

class HierarchicalMechanism(ExponentialMechanism):
    def __init__(self):
        super().__init__()
        self.listHierarchy = None

    def __repr__(self):
        output = super(ExponentialMechanism, self).__repr__()
        output += ".setHierarchy(" + str(self.listHierarchy) + ")" if self.listHierarchy is not None else ""
        
        return output
        
    def __buildHierarchy(self, nestedList, parentNode = []):
        hierarchy = {}

        for _i, _value in enumerate(nestedList):
            if (type(_value) is str):
                hierarchy[_value] = parentNode + [_i]
            elif (type(_value) is not list):
                raise TypeError("All leaves of the hierarchy must be a string " + \
                                 "(see node " + (parentNode + [_i]).__str__() + ")")
            else:
                hierarchy.update(self.__buildHierarchy(_value, parentNode + [_i]))

        self.__checkHierarchyHeight(hierarchy)

        return hierarchy
    
    def __checkHierarchyHeight(self, hierarchy):
        hierarchyHeight = None
        for _value, _hierarchyLocator in hierarchy.items():
            if hierarchyHeight == None:
                hierarchyHeight = len(_hierarchyLocator)
            elif len(_hierarchyLocator) != hierarchyHeight:
                raise ValueError("Leaves of the hierarchy must all be at the same level " +\
                                "(node %s is at level %d instead of hierarchy height %d)" %
                                 (_hierarchyLocator.__str__(), len(_hierarchyLocator), hierarchyHeight))
        return None

    def __buildUtilityList(self, hierarchy):
        if type(hierarchy) is not dict:
            raise TypeError("Hierarchy must be of type dict")
   
        utilityList = []
        hierarchyHeight = None
    
        for _rootValue, _rootHierarchyLocator in hierarchy.items():
            if hierarchyHeight == None:
                hierarchyHeight = len(_rootHierarchyLocator)
                
            for _targetValue, _targetHierarchyLocatior in hierarchy.items():
                if _rootValue >= _targetValue:
                    continue
                
                i = 0
                while (i < len(_rootHierarchyLocator) and
                       _rootHierarchyLocator[i] == _targetHierarchyLocatior[i]):
                    i += 1
                    
                utilityList.append([_rootValue, _targetValue, hierarchyHeight - i])
        
        return utilityList
    
    def setHierarchy(self, listHierarchy):
        if self.epsilon == None:
            raise RuntimeError("Epsilon must be set before hierarchy is set")

        if listHierarchy == None:
            return self

        if type(listHierarchy) is not list:
            raise ValueError("Hierarchy must be a list")
            
        self.listHierarchy = listHierarchy
        hierarchy = self.__buildHierarchy(listHierarchy)
        self.setUtility(self.__buildUtilityList(hierarchy))

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

    def setLabels(self, value1, value2):
        if (type(value1) is not str) or (type(value2) is not str):
            raise ValueError("Binary labels must be strings")

        if len(value1) * len(value2) == 0:
            raise ValueError("Binary labels must be non-empty strings")

        if value1 == value2:
            raise ValueError("Binary labels must be different")

        self.value1 = value1
        self.value2 = value2
        return self
    
    def checkInputs(self, value):
        super().checkInputs(value)

        if (self.value1 is None) or (self.value2 is None):
            raise ValueError("Binary labels must be set")

        if type(value) is not str:
            raise ValueError("Value to be randomised must be a string")

        if value not in [self.value1, self.value2]:
            raise ValueError("Value to be randomised is not in the domain")

    def randomise(self, value):
        self.checkInputs(value)

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

    def preTransform(self, value):
        return value
    
    def postTransform(self, value):
        return value

    def setEpsilon(self, epsilon):
        self.parent.setEpsilon(epsilon)
        return self
    
    def randomise(self, value):
        transformedValue = self.preTransform(value)
        noisyValue = self.parent.randomise(transformedValue)
        outputValue = self.postTransform(noisyValue)
        return outputValue

class RoundedInteger(DPTransformer):
    def postTransform(self, value):
        return int(np.round(value))

class StringtoInt(DPTransformer):
    def preTransform(self, value):
        return int(value)

    def postTransform(self, value):
        return str(value)

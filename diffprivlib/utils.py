import abc
import sys
from copy import copy, deepcopy

# Ensure compatibility with Python 2 and 3 when using ABCMeta
if sys.version_info >= (3, 4):
    ABC = abc.ABC
else:
    ABC = abc.ABCMeta(str('ABC'), (), {})


class DPMachine(ABC):
    @abc.abstractmethod
    def randomise(self, value):
        pass

    def copy(self):
        return copy(self)

    def deepcopy(self):
        return deepcopy(self)

    @abc.abstractmethod
    def set_epsilon(self, epsilon):
        pass

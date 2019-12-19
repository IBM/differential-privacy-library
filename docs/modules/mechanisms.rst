:mod:`diffprivlib.mechanisms`
=============================
.. automodule:: diffprivlib.mechanisms

Base classes
-----------------------------
.. autoclass:: DPMachine
   :members:
   :inherited-members:

.. autoclass:: DPMechanism
   :members:
   :inherited-members:

.. autoclass:: TruncationAndFoldingMixin
   :members:
   :inherited-members:

Binary mechanism
-----------------------------
.. autoclass:: Binary
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_bias,get_mse,get_variance

Exponential mechanisms
-----------------------------
.. autoclass:: Exponential
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_bias,get_mse,get_variance,set_epsilon_delta

.. autoclass:: ExponentialHierarchical
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_bias,get_mse,get_variance,set_epsilon_delta,set_utility

Gaussian mechanisms
-----------------------------
.. autoclass:: Gaussian
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,set_epsilon

.. autoclass:: GaussianAnalytic
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,set_epsilon

Geometric mechanisms
-----------------------------
.. autoclass:: Geometric
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_mse,get_variance,set_epsilon_delta

.. autoclass:: GeometricTruncated
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_bias,get_mse,get_variance,set_epsilon_delta

.. autoclass:: GeometricFolded
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_bias,get_mse,get_variance,set_epsilon_delta

Laplace mechanisms
-----------------------------
.. autoclass:: Laplace
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy

.. autoclass:: LaplaceTruncated
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy

.. autoclass:: LaplaceBoundedDomain
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy

.. autoclass:: LaplaceBoundedNoise
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_mse,get_variance,set_epsilon

.. autoclass:: LaplaceFolded
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_mse,get_variance

Staircase mechanism
-----------------------------
.. autoclass:: Staircase
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_mse,get_variance,set_epsilon_delta

Uniform mechanism
-----------------------------
.. autoclass:: Uniform
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_mse,get_variance,set_epsilon

Vector mechanism
-----------------------------
.. autoclass:: Vector
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_bias,get_mse,get_variance,set_epsilon_delta

Wishart mechanism
-----------------------------
.. autoclass:: Wishart
   :members:
   :inherited-members:
   :exclude-members: copy,deepcopy,get_bias,get_mse,get_variance,set_epsilon_delta

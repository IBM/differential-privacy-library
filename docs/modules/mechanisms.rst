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
   :exclude-members: copy,bias,mse,variance

Bingham mechanism
-----------------------------
.. autoclass:: Bingham
   :members:
   :inherited-members:
   :exclude-members: copy,bias,mse,variance

Exponential mechanisms
-----------------------------
.. autoclass:: Exponential
   :members:
   :inherited-members:
   :exclude-members: copy,bias,mse,variance

.. autoclass:: ExponentialCategorical
   :members:
   :inherited-members:
   :exclude-members: copy,bias,mse,variance,utility_list

.. autoclass:: ExponentialHierarchical
   :members:
   :inherited-members:
   :exclude-members: copy,bias,mse,variance,utility_list

.. autoclass:: PermuteAndFlip
   :members:
   :inherited-members:
   :exclude-members: copy,bias,mse,variance

Gaussian mechanisms
-----------------------------
.. autoclass:: Gaussian
   :members:
   :inherited-members:
   :exclude-members: copy

.. autoclass:: GaussianAnalytic
   :members:
   :inherited-members:
   :exclude-members: copy

.. autoclass:: GaussianDiscrete
   :members:
   :inherited-members:
   :exclude-members: copy,variance,mse

Geometric mechanisms
-----------------------------
.. autoclass:: Geometric
   :members:
   :inherited-members:
   :exclude-members: copy

.. autoclass:: GeometricTruncated
   :members:
   :inherited-members:
   :exclude-members: copy,bias,mse,variance

.. autoclass:: GeometricFolded
   :members:
   :inherited-members:
   :exclude-members: copy,bias,mse,variance

Laplace mechanisms
-----------------------------
.. autoclass:: Laplace
   :members:
   :inherited-members:
   :exclude-members: copy

.. autoclass:: LaplaceTruncated
   :members:
   :inherited-members:
   :exclude-members: copy

.. autoclass:: LaplaceBoundedDomain
   :members:
   :inherited-members:
   :exclude-members: copy

.. autoclass:: LaplaceBoundedNoise
   :members:
   :inherited-members:
   :exclude-members: copy,mse,variance

.. autoclass:: LaplaceFolded
   :members:
   :inherited-members:
   :exclude-members: copy,mse,variance

Snapping mechanism
-----------------------------
.. autoclass:: Snapping
   :members:
   :inherited-members:
   :exclude-members: copy,mse,bias,variance

Staircase mechanism
-----------------------------
.. autoclass:: Staircase
   :members:
   :inherited-members:
   :exclude-members: copy,mse,variance

Uniform mechanism
-----------------------------
.. autoclass:: Uniform
   :members:
   :inherited-members:
   :exclude-members: copy,mse,variance

Vector mechanism
-----------------------------
.. autoclass:: Vector
   :members:
   :inherited-members:
   :exclude-members: copy,bias,mse,variance

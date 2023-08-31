:mod:`diffprivlib.models`
===============================
.. automodule:: diffprivlib.models


Classification models
-------------------------------

Gaussian Naive Bayes
++++++++++++++++++++
.. autoclass:: GaussianNB
   :members:
   :inherited-members:

Logistic Regression
+++++++++++++++++++
.. autoclass:: LogisticRegression
   :members:
   :inherited-members:

Tree-Based Models
+++++++++++++++++++
.. autoclass:: RandomForestClassifier
   :members:
   :inherited-members:
   :exclude-members: feature_importances_,n_features_,base_estimator_

.. autoclass:: DecisionTreeClassifier
   :members:
   :inherited-members:
   :exclude-members: cost_complexity_pruning_path,feature_importances_,n_features_

Regression models
-----------------

Linear Regression
+++++++++++++++++

.. autoclass:: LinearRegression
   :members:
   :inherited-members:


Clustering models
-------------------------------

K-Means
+++++++
.. autoclass:: KMeans
   :members:
   :inherited-members:

Dimensionality reduction models
-------------------------------

PCA
+++
.. autoclass:: PCA
   :members:
   :inherited-members:

Preprocessing
-------------

Standard Scaler
+++++++++++++++
.. autoclass:: StandardScaler
   :members:
   :inherited-members:

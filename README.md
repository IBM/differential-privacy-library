# IBM Differential Privacy Library

[![Python versions](https://img.shields.io/pypi/pyversions/diffprivlib.svg)](https://pypi.org/project/diffprivlib/) [![PyPi version](https://img.shields.io/pypi/v/diffprivlib.svg)](https://pypi.org/project/diffprivlib/) [![Build Status](https://travis-ci.org/IBM/differential-privacy-library.svg?branch=master)](https://travis-ci.org/IBM/differential-privacy-library) [![Documentation Status](https://readthedocs.org/projects/diffprivlib/badge/?version=latest)](https://diffprivlib.readthedocs.io/en/latest/?badge=latest) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/IBM/differential-privacy-library.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/IBM/differential-privacy-library/context:python)

## You have just found the IBM Differential Privacy Library

The IBM Differential Privacy Library is a general-purpose library for experimenting, investigating and developing applications in differential privacy.

Use the Differential Privacy Library if you are looking to:

- Experiment with differential privacy
- Explore the impact of differential privacy on machine learning accuracy using basic classification and clustering models 
- Build your own differential privacy applications, using our extensive collection of mechanisms

Diffprivlib is compatible with: __Python 3.4–3.8__.

## Getting started: [ML with differential privacy in 30 seconds](https://github.com/IBM/differential-privacy-library/blob/master/notebooks/30seconds.ipynb)
We're using the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), so let's load it and perform an 80/20 train/test split.

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

dataset = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
```

Now, let's train a differentially private naive Bayes classifier. Our classifier __runs just like an `sklearn` classifier__, so you can get up and running quickly.

`diffprivlib.models.GaussianNB` can be run __without any parameters__, although this will throw a warning (we need to specify the `bounds` parameter to avoid this). The privacy level is controlled by the parameter `epsilon`, which is passed to the classifier at initialisation (e.g. `GaussianNB(epsilon=0.1)`). The default is `epsilon = 1.0`.

```python
import diffprivlib.models as models

clf = models.GaussianNB()
clf.fit(X_train, y_train)
```

We can now classify unseen examples, knowing that the trained model is differentially private and preserves the privacy of the 'individuals' in the training set (flowers are entitled to their privacy too!).

```python
clf.predict(X_test)
```

Every time the model is trained with `.fit()`, a different model is produced due to the randomness of differential privacy. The accuracy will therefore change, even if it's re-trained with the same training data. Try it for yourself to find out!

```python
from sklearn.metrics import accuracy_score

print("Test accuracy: %f" % accuracy_score(y_test, clf.predict(X_test)))
```

We can easily evaluate the accuracy of the model for various `epsilon` values and plot it with `matplotlib`.

```python
import numpy as np
import matplotlib.pyplot as plt

epsilons = np.logspace(-2, 2, 50)
bounds = [(4.3, 7.9), (2.0, 4.4), (1.1, 6.9), (0.1, 2.5)]
accuracy = list()

for epsilon in epsilons:
    clf = models.GaussianNB(bounds=bounds, epsilon=epsilon)
    clf.fit(X_train, y_train)
    
    accuracy.append(accuracy_score(y_test, clf.predict(X_test)))

plt.semilogx(epsilons, accuracy)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()
```

![Differentially private naive Bayes](https://github.com/IBM/differential-privacy-library/raw/master/notebooks/30seconds.png)

Congratulations, you've completed your first differentially private machine learning task with the Differential Privacy Library!  Check out more examples in the [notebooks](https://github.com/IBM/differential-privacy-library/blob/master/notebooks/) directory, or [dive straight in](https://github.com/IBM/differential-privacy-library/blob/master/diffprivlib/).

## Contents

Diffprivlib is comprised of three modules:
1. __Mechanisms:__ These are the building blocks of differential privacy, and are used in all models that implement differential privacy. Mechanisms have little or no default settings, and are intended for use by experts implementing their own models. They can, however, be used outside models for separate investigations, etc.
1. __Models:__ This module includes machine learning models with differential privacy. Diffprivlib currently has models for clustering, classification, regression, dimensionality reduction and pre-processing.
1. __Tools:__ Diffprivlib comes with a number of generic tools for differentially private data analysis. This includes differentially private histograms, following the same format as [Numpy's histogram function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html).


## Setup

### Installation with `pip`

The library is designed to run with Python 3.
The library can be installed from the PyPi repository using `pip` (or `pip3`):

```bash
pip install diffprivlib
```

### Manual installation

For the most recent version of the library, either download the source code or clone the repository in your directory of choice:

```bash
git clone https://github.com/IBM/differential-privacy-library
```

To install `diffprivlib`, do the following in the project folder (alternatively, you can run `python3 -m pip install .`):
```bash
pip install .
```

The library comes with a basic set of unit tests for `pytest`. To check your install, you can run all the unit tests by calling `pytest` in the install folder:

```bash
pytest
```

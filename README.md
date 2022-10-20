# Diffprivlib v0.6

[![Python versions](https://img.shields.io/pypi/pyversions/diffprivlib.svg)](https://pypi.org/project/diffprivlib/)
[![Downloads](https://pepy.tech/badge/diffprivlib)](https://pepy.tech/project/diffprivlib)
[![PyPi version](https://img.shields.io/pypi/v/diffprivlib.svg)](https://pypi.org/project/diffprivlib/)
[![PyPi status](https://img.shields.io/pypi/status/diffprivlib.svg)](https://pypi.org/project/diffprivlib/)
[![General tests](https://github.com/IBM/differential-privacy-library/actions/workflows/general.yml/badge.svg)](https://github.com/IBM/differential-privacy-library/actions/workflows/general.yml)
[![Documentation Status](https://readthedocs.org/projects/diffprivlib/badge/?version=latest)](https://diffprivlib.readthedocs.io/en/latest/?badge=latest)
[![CodeQL](https://github.com/IBM/differential-privacy-library/actions/workflows/codeql.yml/badge.svg)](https://github.com/IBM/differential-privacy-library/actions/workflows/codeql.yml)
[![codecov](https://codecov.io/gh/IBM/differential-privacy-library/branch/main/graph/badge.svg)](https://codecov.io/gh/IBM/differential-privacy-library)

Diffprivlib is a general-purpose library for experimenting with, investigating and developing applications in, differential privacy.

Use diffprivlib if you are looking to:

- Experiment with differential privacy
- Explore the impact of differential privacy on machine learning accuracy using classification and clustering models 
- Build your own differential privacy applications, using our extensive collection of mechanisms

Diffprivlib supports Python versions 3.8 to 3.10.

## Getting started: [Machine learning with differential privacy in 30 seconds](https://github.com/IBM/differential-privacy-library/blob/main/notebooks/30seconds.ipynb)
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
from diffprivlib.models import GaussianNB

clf = GaussianNB()
clf.fit(X_train, y_train)
```

We can now classify unseen examples, knowing that the trained model is differentially private and preserves the privacy of the 'individuals' in the training set (flowers are entitled to their privacy too!).

```python
clf.predict(X_test)
```

Every time the model is trained with `.fit()`, a different model is produced due to the randomness of differential privacy. The accuracy will therefore change, even if it's re-trained with the same training data. Try it for yourself to find out!

```python
print("Test accuracy: %f" % clf.score(X_test, y_test))
```

We can easily evaluate the accuracy of the model for various `epsilon` values and plot it with `matplotlib`.

```python
import numpy as np
import matplotlib.pyplot as plt

epsilons = np.logspace(-2, 2, 50)
bounds = ([4.3, 2.0, 1.1, 0.1], [7.9, 4.4, 6.9, 2.5])
accuracy = list()

for epsilon in epsilons:
    clf = GaussianNB(bounds=bounds, epsilon=epsilon)
    clf.fit(X_train, y_train)
    
    accuracy.append(clf.score(X_test, y_test))

plt.semilogx(epsilons, accuracy)
plt.title("Differentially private Naive Bayes accuracy")
plt.xlabel("epsilon")
plt.ylabel("Accuracy")
plt.show()
```

![Differentially private naive Bayes](https://github.com/IBM/differential-privacy-library/raw/main/notebooks/30seconds.png)

Congratulations, you've completed your first differentially private machine learning task with the Differential Privacy Library!  Check out more examples in the [notebooks](https://github.com/IBM/differential-privacy-library/blob/main/notebooks/) directory, or [dive straight in](https://github.com/IBM/differential-privacy-library/blob/main/diffprivlib/).

## Contents

Diffprivlib is comprised of four major components:
1. __Mechanisms:__ These are the building blocks of differential privacy, and are used in all models that implement differential privacy. Mechanisms have little or no default settings, and are intended for use by experts implementing their own models. They can, however, be used outside models for separate investigations, etc.
1. __Models:__ This module includes machine learning models with differential privacy. Diffprivlib currently has models for clustering, classification, regression, dimensionality reduction and pre-processing.
1. __Tools:__ Diffprivlib comes with a number of generic tools for differentially private data analysis. This includes differentially private histograms, following the same format as [Numpy's histogram function](https://docs.scipy.org/doc/numpy/reference/generated/numpy.histogram.html).
1. __Accountant:__ The `BudgetAccountant` class can be used to track privacy budget and calculate total privacy loss using advanced composition techniques. 


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

## Citing diffprivlib
If you use diffprivlib for research, please consider citing the following reference paper:
```
@article{diffprivlib,
  title={Diffprivlib: the {IBM} differential privacy library},
  author={Holohan, Naoise and Braghin, Stefano and Mac Aonghusa, P{\'o}l and Levacher, Killian},
  year={2019},
  journal = {ArXiv e-prints},
  archivePrefix = "arXiv",
  volume = {1907.02444 [cs.CR]},
  primaryClass = "cs.CR",
  month = jul
}
```

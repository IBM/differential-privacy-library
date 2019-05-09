# IBM Differential Privacy Library

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/) [![Build Status](https://travis.ibm.com/Naoise-Holohan/ibm-diff-priv-lib.svg?token=5fyN2Bv5EqM4nzxrLe6G&branch=master)](https://travis.ibm.com/Naoise-Holohan/ibm-diff-priv-lib) [![PEP8](https://img.shields.io/badge/code%20style-pep8-orange.svg)](https://www.python.org/dev/peps/pep-0008/)

## You have just found the IBM Differential Privacy Library

The IBM Differential Privacy Library is a general-purpose library for experimenting, investigating and developing applications in differential privacy.

Use the Differential Privacy Library if you are looking to:

- Experiment with differential privacy
- Explore the impact of differential privacy on machine learning accuracy using basic classification, regression and clustering models 
- Build your own differential privacy applications, using our extensive collection of mechanisms

DPL is compatible with: __Python 3.4–3.6__.

## Guiding principles

- __Wide audience:__ DPL is as useful to experts in differential privacy looking to build their own models, as it is to researchers experiencing and experimenting with differential privacy for the first time. Models and tools copy the form of the popular Numpy and SkLearn packages, allowing for basic usage without the need to set any privacy-specific parameters.
- __Extensibility:__ DPL comes with an extensive list of mechanisms, allowing for new and custom models to be written using a common codebase. This will allow for the development of a one-stop-shop for differential privacy.
- __Modularity:__

##Getting started: Differential privacy in 30 seconds

```python
from diffprivlib.models import GaussianNB
import numpy as np

X = np.zeros((10, 3))
y = np.round(np.random(10))
```

To ensure no additional privacy leakage, we must specify the bounds of the data from domain knowledge. If this is not specified, the model will take the bounds from the data and warn you about potential privacy leakage.
We also specify the epsilon value, in this case `1.0`.

```python
epsilon = 1.0
bounds = [(0, 1), (0, 100), (5, 10)]
```

We are now ready to learn a differentially private naive Bayes model on the data.

```python
clf = GaussianNB(epsilon=epsilon, bounds=bounds)
clf.fit(X, y)
```


## Setup

### Manual installation

For the most recent version of the library, either download the source code or clone the repository in your directory of choice:

```bash
git clone https://github.ibm.com/Naoise-Holohan/ibm-diff-priv-lib
```

To install DPL, do the following in the project folder:
```bash
pip install .
```

The library comes with a basic set of unit tests for `pytest`. To check your install, you can run all the unit tests by calling `pytest` in the install folder:

```bash
pytest
```

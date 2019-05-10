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

- __Unifying codebase:__ DPL is the first library of its kind to include a large collection of differential privacy mechanisms, tools and machine learning models. This unifying foundation will make it easier to build new models, tools and mechanisms and to experiment new ways to doing differential privacy.
- __Wide audience:__ DPL is as useful to experts in differential privacy looking to build their own models, as it is to researchers experiencing and experimenting with differential privacy for the first time. Models and tools copy the form of the popular Numpy and SkLearn packages, allowing for basic usage without the need to set any privacy-specific parameters.
- __Extensibility:__ DPL comes with an extensive list of mechanisms, allowing for new and custom models to be written using a common codebase. This will allow for the development of a one-stop-shop for differential privacy.

## Getting started: [ML with differential privacy in 30 seconds](notebooks/30seconds.ipynb)
Let's import `diffprivlib` and other handy functions to get started.

```python
import diffprivlib as dpl
from sklearn import datasets
from sklearn.model_selection import train_test_split
```

We're using the [Iris dataset](https://archive.ics.uci.edu/ml/datasets/iris), so let's load it and perform an 80/20 train/test split.

```python
dataset = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, test_size=0.2)
```

Now, let's train a differentially private naive Bayes classifier and test its accuracy. You'll notice that our classifier runs like an `sklearn` classifier.

`dpl.models.GaussianNB` can be run __without any parameters__, but this will throw a warning (we need to specify `bounds` to fix this). The privacy level is controlled by the parameter `epsilon`, which is passed to the classifier at initialisation (e.g. `dpl.models.GaussianNB(epsilon=0.1)`). The default is `epsilon = 1.0`.

```python
clf = dpl.models.GaussianNB()
clf.fit(X_train, y_train)
```

We can now classify unseen examples, knowing that the trained model is differentially private and preserves the privacy of the 'individuals' in the training set (flowers are entitled to their privacy too!).

```python
clf.predict(X_test)
```

The accuracy of the model will change if the model is re-trained with the same training data. This is due to the randomness of differential privacy. Try it for yourself to find out!

```python
(clf.predict(X_test) == y_test).sum() / y_test.shape[0]
```

Congratulations! You've completed your first differentially private machine learning task with the Differential Privacy Library!  Check out more examples in the [notebooks](notebooks/) directory, or [dive straight in](diffprivlib/).


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

from sklearn.base import BaseEstimator


class KMeans(BaseEstimator):
    def __init__(self, epsilon, n_clusters=8, verbose=0):
        self.epsilon = epsilon
        self.n_clusters = n_clusters
        self.verbose = verbose

    def fit(self, X, y=None, sample_weight=None):
        pass

    def fit_predict(self, X, y=None, sample_weight=None):
        pass

    def predict(self, X, sample_weight=None):
        pass

    def _init_centers(self):
        pass

import numpy as np

from . import DPTransformer


class RoundedInteger(DPTransformer):
    def post_transform(self, value):
        return int(np.round(value))

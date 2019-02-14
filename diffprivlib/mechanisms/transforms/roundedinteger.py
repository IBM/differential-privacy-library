from numpy import round

from . import DPTransformer


class RoundedInteger(DPTransformer):
    def post_transform(self, value):
        return int(round(value))

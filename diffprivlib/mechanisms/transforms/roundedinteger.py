"""
Rounded integer transformer. Rounds the output of the given mechanism to the nearest integer.
"""
from diffprivlib.mechanisms.transforms.base import DPTransformer


class RoundedInteger(DPTransformer):
    """
    Rounded integer transform. Rounds the (float) output of the given mechanism to the nearest integer.
    """
    def post_transform(self, value):
        """
        Transforms the mechanism output to the nearest integer.

        :param value: Mechanism output value to be transformed.
        :type value: `float`
        :return: Transformed mechanism output.
        :rtype: `int`
        """
        return int(round(value))

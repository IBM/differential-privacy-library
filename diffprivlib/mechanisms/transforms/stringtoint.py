"""
StringToInt DP transformer, for using string-valued data with integer-valued mechanisms.
"""
from . import DPTransformer


class StringToInt(DPTransformer):
    """
    StringToInt DP transformer, for using string-valued data with integer-valued mechanisms.

    Useful when using ordered, string-valued data with :class:`.Geometric`.
    """

    def pre_transform(self, value):
        """
        Transforms the input data to be ingested by the differential privacy mechanism.

        :param value: Input value to be transformed.
        :type value: `string`
        :return: Transformed input value.
        :rtype: `int`
        """
        return int(value)

    def post_transform(self, value):
        """
        Transforms the output of the differential privacy mechanism to resemble the input data.

        :param value: Mechanism output to be transformed.
        :type value: `int`
        :return: Transformed output value.
        :rtype: `string`
        """
        return str(value)

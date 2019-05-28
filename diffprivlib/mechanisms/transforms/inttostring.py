"""
IntToString DP transformer, for using integer-valued data with string-valued mechanisms.
"""
from diffprivlib.mechanisms.transforms.base import DPTransformer


class IntToString(DPTransformer):
    """
    IntToString DP transformer, for using integer-valued data with string-valued mechanisms.

    Useful when using integer-valued data with :class:`.Binary` or :class:`.Exponential`.
    """
    def pre_transform(self, value):
        """
        Transforms the input data to be ingested by the differential privacy mechanism.

        :param value: Input value to be transformed.
        :type value: `int`
        :return: Transformed input value.
        :rtype: `string`
        """
        return str(value)

    def post_transform(self, value):
        """
        Transforms the output of the differential privacy mechanism to resemble the input data.

        :param value: Mechanism output to be transformed.
        :type value: `string`
        :return: Transformed output value.
        :rtype: `int`
        """
        return int(value)

from . import DPTransformer


class StringToInt(DPTransformer):
    def pre_transform(self, value):
        return int(value)

    def post_transform(self, value):
        return str(value)
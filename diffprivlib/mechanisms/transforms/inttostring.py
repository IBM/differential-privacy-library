from diffprivlib.mechanisms.transforms import DPTransformer


class IntToString(DPTransformer):
    def pre_transform(self, value):
        return str(value)

    def post_transform(self, value):
        return int(value)

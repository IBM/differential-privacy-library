from diffprivlib import DPMachine


class DPTransformer(DPMachine):
    def __init__(self, parent):
        if not isinstance(parent, DPMachine):
            raise TypeError("Data transformer must take a DPMachine as input")

        self.parent = parent

    def pre_transform(self, value):
        return value

    def post_transform(self, value):
        return value

    def set_epsilon(self, epsilon):
        self.parent.set_epsilon(epsilon)
        return self

    def randomise(self, value):
        transformed_value = self.pre_transform(value)
        noisy_value = self.parent.randomise(transformed_value)
        output_value = self.post_transform(noisy_value)
        return output_value

class Tensor:
    def __init__(self, name, shape):
        self.name = name
        self.shape = shape
        self.dtype = "f32"
class Add:
    def __init__(self, lhs: Tensor, rhs: Tensor, result: Tensor):
        self.lhs = lhs
        self.rhs = rhs
        self.result = result

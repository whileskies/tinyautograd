
class SGD:
    def __init__(self, parameters, lr=0.01):
        self._parameters = parameters
        self.lr = lr

    def step(self):
        for p in self._parameters:
            if p._grad is not None:
                p._data -= self.lr * p._grad

    def zero_grad(self):
        for p in self._parameters:
            p._grad = None
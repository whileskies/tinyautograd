from .tensor import Tensor
from .functional import relu
import numpy as np

class Layer:
    def parameters(self): return []

class Linear(Layer):
    def __init__(self, in_features, out_features):
        self.W = Tensor(np.random.randn(in_features, out_features), requires_grad=True, label=f'Linear[{in_features}, {out_features}] W')
        self.b = Tensor(np.zeros((1, out_features)), requires_grad=True, label=f'Linear[{in_features}, {out_features}] b')
        # print(self.W.data)
    
    def __call__(self, x: Tensor):
        return x.matmul(self.W) + self.b
    
    def parameters(self):
        return [self.W, self.b]


class MLP(Layer):
    def  __init__(self, in_dim, hiddens, out_dim):
        sizes = [in_dim] + hiddens + [out_dim]
        self.layers = [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
    
    def __call__(self, x: Tensor):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = relu(x)
        return x
    
    def parameters(self):
        return [p for l in self.layers for p in l.parameters()]



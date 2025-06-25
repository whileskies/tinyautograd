from .tensor import Tensor
from .functional import relu
import numpy as np

class Layer:
    def parameters(self): return []

class Linear(Layer):
    def __init__(self, in_features, out_features):
        # ReLU 非线性会让一部分神经元"死亡"(输出恒为0)，使用He初始化(Kaiming Init)
        self._W = Tensor(np.random.randn(in_features, out_features) * np.sqrt(2 / in_features), requires_grad=True, label=f'Linear[{in_features}, {out_features}] W')
        self._b = Tensor(np.zeros((1, out_features)), requires_grad=True, label=f'Linear[{in_features}, {out_features}] b')
    
    def __call__(self, x: Tensor):
        return x.matmul(self._W) + self._b
    
    def parameters(self):
        return [self._W, self._b]


class MLP(Layer):
    def  __init__(self, in_dim, hiddens, out_dim, activation_fun=relu):
        sizes = [in_dim] + hiddens + [out_dim]
        self._layers = [Linear(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]
        self._activation_fun = activation_fun
    
    def __call__(self, x: Tensor):
        for i, layer in enumerate(self._layers):
            x = layer(x)
            if i < len(self._layers) - 1:
                x = self._activation_fun(x)
        return x
    
    def parameters(self):
        return [p for l in self._layers for p in l.parameters()]



from .nn import Linear
from .nn import MLP
from .tensor import Tensor

def test_layer():
    L = Linear(5, 3)
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    y = L(x)
    print(L.parameters())
    print(y.data)

def test_mlp():
    M = MLP(5, [10, 10], 3)
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    y = M(x)
    print(y)
    print(y.data)
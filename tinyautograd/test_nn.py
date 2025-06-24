from .nn import Linear
from .nn import MLP
from .tensor import Tensor
from .functional import mse_loss

def assert_grad_right(a, b):
    assert abs(a - b) < 1, "grad wrong"

def test_layer():
    L = Linear(5, 3)
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    y = L(x)
    print(L.parameters())
    print(y.data)

def test_mlp():
    M = MLP(5, [10, 10], 3)
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
    t = Tensor([1, 2, 3])
    y = mse_loss(M(x), t)
    print(y.data)
    y.backward()

    x2 = Tensor([[1.001, 2, 3, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
    y2 = mse_loss(M(x2), t)
    print((y2.data - y.data)/0.001)
    print(x.grad[0][0])
    assert_grad_right((y2.data - y.data)/0.001, x.grad[0][0])
    
def test_loss_desc():
    M = MLP(5, [10, 10], 3)
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    t = Tensor([1, 2, 3], [4, 5, 6])
    print('shape', M(x).shape)
    y = mse_loss(M(x), t)
    print('y', y.data)

    y.backward()
    print(M.parameters())
    for p in M.parameters():
        if p.grad is not None:
            p._data -=  0.0001 * p.grad

    y2 = mse_loss(M(x), t)
    print('y2', y2.data)
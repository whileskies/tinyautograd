from .nn import Linear
from .nn import MLP
from .tensor import Tensor
from .functional import mse_loss, tanh
from .optim import SGD
import numpy as np

def assert_grad_right(a, b):
    assert abs(a - b) < 3, "grad wrong"

def test_layer():
    L = Linear(5, 3)
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    y = L(x)
    print(L.parameters())
    print(y.data)


def test_layer_cuda():
    L = Linear(5, 3)
    L.to('cuda')
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    x.to('cuda')
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
    print(x.grad.npdata)

    x2 = Tensor([[1.001, 2, 3, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
    y2 = mse_loss(M(x2), t)
    g = (y2.data - y.data)/0.001
    print(g)
    assert_grad_right(x.grad.npdata[0][0], g)


def test_mlp_cuda():
    M = MLP(5, [10, 10], 3)
    M.to('cuda')
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
    x.to('cuda')
    t = Tensor([1, 2, 3])
    t.to('cuda')
    y = mse_loss(M(x), t)
    y.backward()
    print(y.data)
    print(x.grad.npdata)


    x2 = Tensor([[1.001, 2, 3, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
    x2.to('cuda')
    y2 = mse_loss(M(x2), t)
    g = (y2.data - y.data)/0.001
    print(g)
    assert_grad_right(x.grad.npdata[0][0], g)


def test_loss_desc():
    for _ in range(100):
        M = MLP(5, [10, 10], 3)
        x = Tensor(np.random.randn(2, 5))          # shape (2, 5)
        t = Tensor(np.random.randn(2, 3))          # shape (2, 3)

        y = mse_loss(M(x), t)
        y.backward()

        for p in M.parameters():
            if p.grad is not None:
                p._data -= 0.0001 * p.grad

        y2 = mse_loss(M(x), t)
        assert y2.data < y.data, f"Loss did not decrease: before={y.data}, after={y2.data}"


def test_loss_desc_cuda():
    for _ in range(100):
        M = MLP(5, [10, 10], 3)
        x = Tensor(np.random.randn(2, 5))
        t = Tensor(np.random.randn(2, 3))

        M.to('cuda')
        x.to('cuda')
        t.to('cuda')

        y = mse_loss(M(x), t)
        y.backward()

        for p in M.parameters():
            if p.grad is not None:
                p._data -= 0.0001 * p.grad

        y2 = mse_loss(M(x), t)
        assert y2.data < y.data, f"Loss did not decrease (CUDA): before={y.data}, after={y2.data}"


def test_sgd():
    for i in range(100):
        M = MLP(5, [10, 10], 3)
        opt = SGD(M.parameters(), lr=0.0001)

        x = Tensor(np.random.randn(2, 5))
        t = Tensor(np.random.randn(2, 3))

        y = mse_loss(M(x), t)
        y.backward()
        opt.step()

        y2 = mse_loss(M(x), t)
        assert y2.data < y.data, f"iter={i}, loss not reduced: before={y.data}, after={y2.data}"


def test_sgd_cuda():
    for i in range(100):
        M = MLP(5, [10, 10], 3)
        opt = SGD(M.parameters(), lr=0.0001)

        x = Tensor(np.random.randn(2, 5))
        t = Tensor(np.random.randn(2, 3))

        M.to('cuda')
        x.to('cuda')
        t.to('cuda')

        y = mse_loss(M(x), t)
        y.backward()
        opt.step()

        y2 = mse_loss(M(x), t)
        assert y2.data < y.data, f"[CUDA] iter={i}, loss not reduced: before={y.data}, after={y2.data}"


def test_train():
    model = MLP(3, [4, 4], 1, activation_fun=tanh)
    opt = SGD(model.parameters(), lr=0.05)

    x = Tensor([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
    ys = Tensor([[1.0], [-1.0], [-1.0], [1.0]])
    print(model(x).data)
    loss = mse_loss(model(x), ys)
    print(loss.data)

    for i in range(500):
        loss = mse_loss(model(x), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(i, loss.data)

    print('pred: ', model(x).data)


def test_train_cuda():
    model = MLP(3, [4, 4], 1, activation_fun=tanh)
    model.to('cuda')
    opt = SGD(model.parameters(), lr=0.05)

    x = Tensor([[2.0, 3.0, -1.0], [3.0, -1.0, 0.5], [0.5, 1.0, 1.0], [1.0, 1.0, -1.0]])
    ys = Tensor([[1.0], [-1.0], [-1.0], [1.0]])
    x.to('cuda')
    ys.to('cuda')
    print(model(x).data)
    loss = mse_loss(model(x), ys)
    print(loss.data)

    for i in range(500):
        loss = mse_loss(model(x), ys)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(i, loss.data)

    print('pred: ', model(x).data)
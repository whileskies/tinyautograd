from .tensor import *
from .functional import *

def test_grad():
    a = Tensor([[1, 1],[2, 2],[3,3]], requires_grad=True)
    b = Tensor([[4, 4],[5, 5],[6, 6]], requires_grad=True)
    c = Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]], requires_grad=True)

    d = ((3*(a * b + c)) ** 2).sum()
    d.backward()
    print(d.data)
    print(a.grad)

    a2 = Tensor([[1, 1], [2, 2.0001], [3, 3]], requires_grad=True)
    d2 = ((3*(a2 * b + c)) ** 2).sum()
    print((d2.data - d.data) / 0.0001)


def test_softmax_cross_entropy():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([0, 1, 0], requires_grad=False)
    c = softmax_cross_entropy(a, b)
    c.backward()
    print(c.data)
    print(a.grad)

    a2 = Tensor([1, 2.001, 3], requires_grad=True)
    c2 = softmax_cross_entropy(a2, b)

    d2 = ((c2.data - c.data) / 0.001)
    print(d2)


def test_mean():
    a = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
    print('aaa')
    b = a.mean()
    b.backward()
    print(b.data)
    print(a.grad)

    a2 = Tensor([[1, 2, 3.001, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
    b2 = a2.mean()

    d = (b2.data - b.data) / 0.001
    print(d)

def test_relu():
    a = Tensor([1], requires_grad=True)
    b = relu(a) + relu(a) + relu(a)
    print(b.data)
    b.backward()

    a2 = Tensor([1.001], requires_grad=True)
    b2 = relu(a2) + relu(a2) + relu(a2)
    d = (b2.data - b.data) / 0.001
    print(d)
    print(a.grad)


def test_mse():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1.1, 2.1, 3.1], requires_grad=True)
    y = mse_loss(a, b)
    print(y.data)


def test_matmul():
    a = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], requires_grad=True)
    b = Tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]], requires_grad=True)
    y = (a.matmul(b)).sum()
    print(y.data)
    y.backward()

    a2 = Tensor([[1.001, 2, 3], [1, 2, 3], [1, 2, 3]], requires_grad=True)
    y2 = (a2.matmul(b)).sum()
    print(y2.data)
    print((y2.data - y.data)/0.001)
    print(a.grad[0][0])
    assert a.grad[0][0] - y2.data < 1e-2
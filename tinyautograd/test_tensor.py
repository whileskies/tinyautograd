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
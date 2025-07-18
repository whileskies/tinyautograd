from .tensor import *
from .functional import *

def assert_grad_right(a, b):
    assert abs(a - b) < 1, "grad wrong"


def test_add():
    a = Tensor([[1, 1],[2, 2],[3,3]], requires_grad=True)
    b = Tensor([[4, 4],[5, 5],[6, 6]], requires_grad=True)
    c = Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]], requires_grad=True)

    d = (a + b + c).sum()
    print(d.data)
    print(d.shape)
    d.backward()
    print(a.grad.npdata)

    a2 = Tensor([[1, 1], [2, 2.001], [3, 3]], requires_grad=True)
    d2 = (a2 + b + c).sum()
    print(d2.data)
    g = (d2.data - d.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[1][1], g)


def test_add_cuda():
    a = Tensor([[1, 1],[2, 2],[3,3]], requires_grad=True)
    b = Tensor([[4, 4],[5, 5],[6, 6]], requires_grad=True)
    c = Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]], requires_grad=True)
    a.to('cuda')
    b.to('cuda')
    c.to('cuda')
    d = (a + b + c).sum()
    print(d.data)
    print(d.shape)
    d.backward()
    print(a.grad.npdata)

    a2 = Tensor([[1, 1], [2, 2.001], [3, 3]], requires_grad=True)
    a2.to('cuda')
    d2 = (a2 + b + c).sum()
    print(d2.data)
    g = (d2.data - d.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[1][1], g)
    

def test_mul():
    a = Tensor([[1, 1],[2, 2],[3,3]], requires_grad=True)
    b = Tensor([[4, 4],[5, 5],[6, 6]], requires_grad=True)
    c = Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]], requires_grad=True)

    d = (3*(a * b + c)).sum()
    d.backward()
    print(d.data)
    print(a.grad.npdata)

    a2 = Tensor([[1, 1], [2, 2.0001], [3, 3]], requires_grad=True)
    d2 = (3*(a2 * b + c)).sum()
    g = (d2.data - d.data) / 0.0001
    print(g)
    assert_grad_right(a.grad.npdata[1][1], g)


def test_mul_cuda():
    a = Tensor([[1, 1],[2, 2],[3,3]], requires_grad=True)
    b = Tensor([[4, 4],[5, 5],[6, 6]], requires_grad=True)
    c = Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]], requires_grad=True)
    a.to('cuda')
    b.to('cuda')
    c.to('cuda')

    d = (3*(a * b + c)).sum()
    d.backward()
    print(d.data)
    print(a.grad.npdata)

    a2 = Tensor([[1.001, 1], [2, 2], [3, 3]], requires_grad=True)
    a2.to('cuda')
    d2 = (3*(a2 * b + c)).sum()
    g = (d2.data - d.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0][0], g)


def test_power():
    a = Tensor([[1, 1],[2, 2],[3,3]], requires_grad=True)
    b = Tensor([[4, 4],[5, 5],[6, 6]], requires_grad=True)
    c = Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]], requires_grad=True)

    d = ((3*(a * b + c))**2).sum()
    d.backward()
    print(d.data)
    print(a.grad.npdata)

    a2 = Tensor([[1, 1], [2, 2.001], [3, 3]], requires_grad=True)
    d2 = ((3*(a2 * b + c))**2).sum()
    g = (d2.data - d.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[1][1], g)


def test_power_cuda():
    a = Tensor([[1, 1],[2, 2],[3,3]], requires_grad=True)
    b = Tensor([[4, 4],[5, 5],[6, 6]], requires_grad=True)
    c = Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]], requires_grad=True)
    a.to('cuda')
    b.to('cuda')
    c.to('cuda')

    d = ((3*(a * b + c))**2).sum()
    d.backward()
    print(d.data)
    print(a.grad.npdata)

    a2 = Tensor([[1, 1], [2.001, 2], [3, 3]], requires_grad=True)
    a2.to('cuda')
    d2 = ((3*(a2 * b + c))**2).sum()
    g = (d2.data - d.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[1][0], g)

# def test_softmax_cross_entropy():
#     a = Tensor([1, 2, 3], requires_grad=True)
#     b = Tensor([0, 1, 0], requires_grad=False)
#     c = softmax_cross_entropy(a, b)
#     c.backward()
#     print(c.data)
#     print(a.grad)

#     a2 = Tensor([1, 2.001, 3], requires_grad=True)
#     c2 = softmax_cross_entropy(a2, b)

#     d2 = ((c2.data - c.data) / 0.001)
#     print(d2)


# def test_mean():
#     a = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
#     print('aaa')
#     b = a.mean()
#     b.backward()
#     print(b.data)
#     print(a.grad)

#     a2 = Tensor([[1, 2, 3.001, 4, 5], [2, 4, 6, 8, 10]], requires_grad=True)
#     b2 = a2.mean()

#     d = (b2.data - b.data) / 0.001
#     print(d)

# def test_relu():
#     a = Tensor([1], requires_grad=True)
#     b = relu(a) + relu(a) + relu(a)
#     print(b.data)
#     b.backward()

#     a2 = Tensor([1.001], requires_grad=True)
#     b2 = relu(a2) + relu(a2) + relu(a2)
#     d = (b2.data - b.data) / 0.001
#     print(d)
#     print(a.grad)


def test_mse():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1.2, 3, 3.5], requires_grad=True)
    y = mse_loss(a, b)
    y.backward()
    print(y.data)
    print(a.grad.npdata)

    a2 = Tensor([1.001, 2, 3], requires_grad=True)
    y2 = mse_loss(a2, b)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0], g)


def test_mse_cuda():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([1.2, 3, 3.5], requires_grad=True)
    a.to('cuda')
    b.to('cuda')
    y = mse_loss(a, b)
    y.backward()
    print(y.data)
    print(b.grad.npdata)

    b2 = Tensor([1.2, 3.001, 3.5], requires_grad=True)
    b2.to('cuda')
    y2 = mse_loss(a, b2)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(b.grad.npdata[1], g)

def test_matmul():
    a = Tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]], requires_grad=True)
    b = Tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]], requires_grad=True)
    y = (a.matmul(b)).sum()
    y.backward()
    print(y.data)
    print(a.grad.npdata)
    

    a2 = Tensor([[1.001, 2, 3], [1, 2, 3], [1, 2, 3]], requires_grad=True)
    y2 = (a2.matmul(b)).sum()
    print(y2.data)
    g = (y2.data - y.data)/0.001
    print(g)
    assert_grad_right(a.grad.npdata[0][0], g)
    # print((y2.data - y.data)/0.001)
    # print(a.grad[0][0])
    # assert a.grad[0][0] - y2.data < 1e-2


def test_matmul_cuda():
    a = Tensor([[1, 2, 3], [1, 2, 3]], requires_grad=True)
    b = Tensor([[4, 5], [7, 8], [1, 2]], requires_grad=True)
    a.to('cuda')
    b.to('cuda')
    y = (a.matmul(b)).sum()
    y.backward()
    print(y.data)
    print(a.grad.npdata)
    

    a2 = Tensor([[1, 2.001, 3], [1, 2, 3]], requires_grad=True)
    a2.to('cuda')
    y2 = (a2.matmul(b)).sum()
    print(y2.data)
    g = (y2.data - y.data)/0.001
    print(g)
    assert_grad_right(a.grad.npdata[0][1], g)


def test_relu():
    a = Tensor([1, -2, 2, 3, -1, -5], requires_grad=True)
    y = (relu(a) + relu(a) + relu(a)).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([1.001, -2, 2, 3, -1, -5], requires_grad=True)
    y2 = (relu(a2) + relu(a2) + relu(a2)).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0], g)


def test_relu_cuda():
    a = Tensor([1, -2, 2, 3, -1, -5], requires_grad=True)
    y = (relu(a) + relu(a) + relu(a)).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([1.001, -2, 2, 3, -1, -5], requires_grad=True)
    y2 = (relu(a2) + relu(a2) + relu(a2)).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0], g)


def test_log():
    a = Tensor([0.5, 0.8, 1, 2, 4, 5], requires_grad=True)
    y = (log(a) + log(a) + log(a)).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([0.5, 0.801, 1, 2, 4, 5], requires_grad=True)
    y2 = (log(a2) + log(a2) + log(a2)).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[1], g)


def test_log_cuda():
    a = Tensor([0.5, 0.8, 1, 2, 4, 5], requires_grad=True)
    a.to('cuda')
    y = (log(a) + log(a) + log(a)).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([0.5, 0.801, 1, 2, 4, 5], requires_grad=True)
    a2.to('cuda')
    y2 = (log(a2) + log(a2) + log(a2)).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[1], g)


def test_tanh():
    a = Tensor([0.1, 0.2, 0.3], requires_grad=True)
    y = (tanh(a) + tanh(a) + tanh(a)).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([0.101, 0.2, 0.3], requires_grad=True)
    y2 = (tanh(a2) + tanh(a2) + tanh(a2)).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0], g)


def test_tanh_cuda():
    a = Tensor([0.1, 0.2, 0.3], requires_grad=True)
    a.to('cuda')
    y = (tanh(a) + tanh(a) + tanh(a)).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([0.101, 0.2, 0.3], requires_grad=True)
    a2.to('cuda')
    y2 = (tanh(a2) + tanh(a2) + tanh(a2)).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0], g)


def test_sigmod():
    a = Tensor([2, 4, 6, 8], requires_grad=True)
    y = (sigmod(a) + sigmod(a) + sigmod(a)).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([2.001, 4, 6, 8], requires_grad=True)
    y2 = (sigmod(a2) + sigmod(a2) + sigmod(a2)).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0], g)


def test_sigmod_cuda():
    a = Tensor([2, 4, 6, 8], requires_grad=True)
    a.to('cuda')
    y = (sigmod(a) + sigmod(a) + sigmod(a)).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([2, 4.001, 6, 8], requires_grad=True)
    a2.to('cuda')
    y2 = (sigmod(a2) + sigmod(a2) + sigmod(a2)).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[1], g)


def test_softmax_crossentropy():
    a = Tensor([[5, -1, 2, 3, 0, 1], [1, 2, 3, 4, 5, 6]], requires_grad=True)
    b = Tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], requires_grad=True)
    y = softmax_cross_entropy(a, b).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([[5.001, -1, 2, 3, 0, 1], [1, 2, 3, 4, 5, 6]], requires_grad=True)
    y2 = softmax_cross_entropy(a2, b).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0][0], g)



def test_softmax_crossentropy_cuda():
    a = Tensor([[5, -1, 2, 3, 0, 1], [1, 2, 3, 4, 5, 6]], requires_grad=True)
    b = Tensor([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], requires_grad=True)
    a.to('cuda')
    b.to('cuda')
    y = softmax_cross_entropy(a, b).sum()
    print(y.data)
    y.backward()
    print(a.grad.npdata)

    a2 = Tensor([[5, -1, 2.001, 3, 0, 1], [1, 2, 3, 4, 5, 6]], requires_grad=True)
    a2.to('cuda')
    y2 = softmax_cross_entropy(a2, b).sum()
    print(y2.data)
    g = (y2.data - y.data) / 0.001
    print(g)
    assert_grad_right(a.grad.npdata[0][2], g)
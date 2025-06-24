from tinyautograd import *

a = Tensor([[1, 1],[2, 2],[3,3]], requires_grad=True)
b = Tensor([[4, 4],[5, 5],[6, 6]], requires_grad=True)
c = Tensor([[1.1, 1.1], [2.2, 2.2], [3.3, 3.3]], requires_grad=True)

d = ((a * b + c) ** 5).mean()
d.backward()
print(d.data)
print(a.grad)

a2 = Tensor([[1, 1.001], [2, 2], [3, 3]], requires_grad=True)
d2 = ((a2 * b + c) ** 5).mean()
print((d2.data - d.data) / 0.001)

aa = Tensor([1, 2, 3], requires_grad=True)
bb = Tensor([0, 1, 0], requires_grad=False)
cc = softmax_cross_entropy(aa, bb)
cc.backward()
print(cc.data)
print(aa.grad)

aa2 = Tensor([1.001, 2, 3], requires_grad=True)
cc2 = softmax_cross_entropy(aa2, bb)

dd2 = ((cc2.data - cc.data) / 0.001)
print(dd2)

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
    print(a.grad)


def test_loss_desc():
    M = MLP(5, [10, 10], 3)
    x = Tensor([[1, 2, 3, 4, 5], [2, 4, 6, 8, 10]])
    t = Tensor([1, 2, 3], [4, 5, 6])
    print('shape', M(x).shape)
    y = mse_loss(M(x), t)
    print(y)

    y.backward()
    print(M.parameters())
    for p in M.parameters():
        print(p)
        print('data', p.data.shape)
        print('grad', p.grad.shape)
    #     if p.grad is not None:
    #         p.data -= 0.01 * p.grad

test_loss_desc()
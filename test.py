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
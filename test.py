from tinyautograd import *

a = Tensor([1, 2, 3], requires_grad=True)
b = Tensor([4, 5, 6], requires_grad=True)
c = a * b
c.backward()
print(c.data)
print(b.grad)
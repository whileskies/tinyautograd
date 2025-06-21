import numpy as np

class Tensor:
    def __init__(self, data, op='', requires_grad=False, label=''):
        self._data = np.array(data, dtype=np.float32)
        self._label = label
        self._op = op
        self._shape = self._data.shape
        self._grad = None
        self._requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set()
    
    def __repr__(self):
        return f"Tensor(lable={self._label}, op={self._op}, shape={self._shape})"
    
    @property
    def data(self):
        return self._data
    
    @property
    def label(self):
        return self._label
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def grad(self):
        return self._grad

    def _add_grad(self, grad):
        if self._grad is None:
            self._grad = grad
        else:
            self._grad += grad 
    
    def backward(self):
        if self._grad is None:
            self._grad = np.ones_like(self._data)
        
        topo, visited = [], set()

        def build(node):    
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)
    
        build(self)
        print(topo)

        for node in reversed(topo):
            node._backward()


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data + other._data, op='add', requires_grad=self._requires_grad or other._requires_grad)
        out._prev = {self, other}

        def _backward():
            if self._requires_grad:
                self._grad = self._grad + out._grad if self._grad is not None else out._grad
            if other._requires_grad:
                other._grad = other._grad + out._grad if other._grad is not None else out._grad
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data * other._data, op='mul', requires_grad=self._requires_grad or other._requires_grad)
        out._prev = {self, other}

        def _backward():
            if self._requires_grad:
                grad = other._data * out._grad
                self._grad = self._grad + grad if self._grad is not None else grad
            if other._requires_grad:
                grad = self._data * out._grad
                other._grad = other._grad + grad if other._grad is not None else grad

        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supporting int/float powers for now"
        out = Tensor(self._data**power, op='pow', requires_grad=self._requires_grad)
        out._prev = {self}

        def _backward():
            if self._requires_grad:
                grad = power*(self._data**(power-1)) * out._grad
                self._grad = self._grad + grad if self._grad is not None else grad
        
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def matmul(self, other):
        out = Tensor(self._data @ other._data, requires_grad=self._requires_grad or other._requires_grad)
        out._prev = {self, other}

        def _backward():
            if self._requires_grad:
                # print('aa', other._data.T)
                # print('out', out._grad)
                grad = out._grad @ other._data.T
                self._add_grad(grad)
            if other._requires_grad:
                grad = self._data.T @ out._grad
                other._add_grad(grad)
        out._backward = _backward

        return out


    def sum(self):
        out = Tensor(self._data.sum(), op='sum', requires_grad=self._requires_grad)
        out._prev = {self}
        
        def _backward():
            if self._requires_grad:
                grad = np.ones_like(self._data) * out._grad
                self._grad = self._grad + grad if self._grad is not None else grad

        out._backward = _backward

        return out


    def mean(self):
        out = Tensor(self._data.mean(), op='mean', requires_grad=self._requires_grad)
        out._prev = {self}
        
        def _backward():
            if self._requires_grad:
                grad = np.ones_like(self._data) / self._data.size * out._grad
                self._grad = self._grad + grad if self._grad is not None else grad

        out._backward = _backward

        return out
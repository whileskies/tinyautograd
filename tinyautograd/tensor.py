from .rawtensor import *

class Tensor:
    def __init__(self, data, op='', label='', requires_grad=False):
        if isinstance(data, RawTensor):
            self._data = data
        else:
            self._data = RawTensor(data)
        self._op = op
        self._label = label
        self._grad = None
        self._requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set()


    def __repr__(self):
        return f"Tensor(op={self._op}, label={self._label})"
    
    @property
    def data(self):
        return self._data.npdata
    
    # @data.setter
    # def data(self, data):
    #     self._data = data
    
    @property
    def shape(self):
        return self._data.shape
    
    @property
    def grad(self):
        return self._grad
    
    @property
    def size(self):
        return self._data.size
    
    @property
    def device(self):
        return self._data.device

    @property
    def requires_grad(self):
        return self._requires_grad 
    
    
    def to(self, device):
        if device not in ["cpu", "cuda"]:
            raise ValueError("Unsupported device. Choose 'cpu' or 'cuda'.")
        
        self._data.to(device)

        return self


    def _add_grad(self, grad):
        if self._grad is None:
            self._grad = grad
        else:
            self._grad += grad 
    
    def backward(self):
        if self._grad is None:
            self._grad = RawTensor.ones_like(self._data)
        
        topo, visited = [], set()

        def build(node):    
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build(child)
                topo.append(node)
    
        build(self)
        # print(topo)

        for node in reversed(topo):
            node._backward()


    def _unbroadcast(self, grad, shape):
        """将梯度 grad 还原成目标 shape（用于广播反向传播）"""
        while len(shape) < len(grad.shape):
            shape = (1,) + shape
        for i, (g, s) in enumerate(zip(grad.shape, shape)):
            if s == 1:
                grad = grad.sum(axis=i, keepdims=True)
        grad.shape = shape
        return grad


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self._data + other._data, op='add', requires_grad=self._requires_grad or other._requires_grad)
        out._prev = {self, other}

        def _backward():
            if self._requires_grad:
                grad = self._unbroadcast(out._grad, self._data.shape)
                # grad = out._grad
                self._add_grad(grad)
            if other._requires_grad:
                grad = self._unbroadcast(out._grad, other._data.shape)
                # grad = out._grad
                other._add_grad(grad)
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other


    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        other.to(self.device)
        out = Tensor(self._data * other._data, op='mul', requires_grad=self._requires_grad or other._requires_grad)
        out._prev = {self, other}

        def _backward():
            if self._requires_grad:
                grad = other._data * out._grad
                self._add_grad(grad)
            if other._requires_grad:
                grad = self._data * out._grad
                other._add_grad(grad)

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
        out._requires_grad = self._requires_grad

        def _backward():
            if self._requires_grad:
                grad = RawTensor.pow_grad(self._data, power) * out._grad
                self._add_grad(grad)
        
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        other = float(other)
        return self * other**-1
    

    def matmul(self, other):
        out = Tensor(self._data @ other._data, requires_grad=self._requires_grad or other._requires_grad)
        out._prev = {self, other}

        def _backward():
            if self._requires_grad:
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
                grad = RawTensor.ones_like(self._data) * out._grad
                self._add_grad(grad)

        out._backward = _backward

        return out


    def mean(self):
        sz = self._data.size
        out = Tensor(self._data.sum() / sz, requires_grad=self._requires_grad)
        out._prev = {self}
         
        def _backward():
            if self._requires_grad:
                grad = np.ones_like(self._data) / sz * out._grad
                self._add_grad(grad)
        out._backward = _backward

        return out
    
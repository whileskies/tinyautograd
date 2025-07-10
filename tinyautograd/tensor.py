import numpy as np
import ctypes
from .ops import Ops

class Tensor:
    def __init__(self, data, op='', requires_grad=False, label='', device='cpu', shape=None):
        if device == 'cpu':
            self._data = np.array(data, dtype=np.float32)
            self._shape = self._data.shape
            self._device = 'cpu'
        elif device == 'cuda':
            self._data = data
            self._shape = shape
            self._device = 'cuda'
        else:
            raise ValueError(f"不支持的设备类型：{device}")

        self._label = label
        self._op = op
        self._grad = None
        self._requires_grad = requires_grad
        self._backward = lambda: None
        self._prev = set()


    def __repr__(self):
        return f"Tensor(lable={self._label}, op={self._op}, shape={self._shape})"
    
    @property
    def data(self):
        return self._data
    
    @data.setter
    def data(self, data):
        self._data = data
    
    @property
    def label(self):
        return self._label
    
    @property
    def shape(self):
        return self._shape
    
    @property
    def grad(self):
        return self._grad
    
    @property
    def size(self):
        return np.prod(self._shape)
    
    @property
    def device(self):
        return self._device

    @property
    def requires_grad(self):
        return self._requires_grad 
    
    
    
    def to(self, device):
        if device not in ["cpu", "cuda"]:
            raise ValueError("Unsupported device. Choose 'cpu' or 'cuda'.")
        if self._device == device:
            return self
        
        self._data = Ops.to_device(self, device)
        self._device = device

        return self

    
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
        # print(topo)

        for node in reversed(topo):
            node._backward()

    def _unbroadcast(self, grad, shape):
        """将梯度 grad 还原成目标 shape（用于广播反向传播）"""

        # while len(grad.shape) > len(shape):
        #     grad = grad.sum(axis=0)
        # for i, (g_dim, s_dim) in enumerate(zip(grad.shape, shape)):
        #     if s_dim == 1 and g_dim != 1:
        #         grad = grad.sum(axis=i, keepdims=True)
        if grad.shape[0] != shape[0]:
            return np.sum(grad, axis=0, keepdims=True)
        else:
            return grad


    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Ops.add(self, other)
        # out = Tensor(Ops.add(self, other), op='add', requires_grad=self._requires_grad or other._requires_grad)
        out._prev = {self, other}
        out._requires_grad = self._requires_grad or other._requires_grad

        def _backward():
            if self._requires_grad:
                grad = self._unbroadcast(out._grad, self._data.shape)
                self._add_grad(grad)
            if other._requires_grad:
                grad = self._unbroadcast(out._grad, other._data.shape)
                other._add_grad(grad)
        out._backward = _backward

        return out
    
    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Ops.mul(self, other)
        # out = Tensor(self._data * other._data, op='mul', requires_grad=self._requires_grad or other._requires_grad)
        out._prev = {self, other}
        out._requires_grad = self._requires_grad or other._requires_grad

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
        # out = Tensor(self._data**power, op='pow', requires_grad=self._requires_grad)
        out = Ops.pow(self)
        out._prev = {self}
        out._requires_grad = self._requires_grad

        def _backward():
            if self._requires_grad:
                grad = power*(self._data**(power-1)) * out._grad
                self._grad = self._grad + grad if self._grad is not None else grad
        
        out._backward = _backward

        return out
    
    def __truediv__(self, other):
        return self * other**-1
    
    def matmul(self, other):
        # out = Tensor(self._data @ other._data, requires_grad=self._requires_grad or other._requires_grad)
        out = Ops.matmul(self, other)
        out._prev = {self, other}
        out._requires_grad = self._requires_grad or other._requires_grad

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
        # out = Tensor(self._data.sum(), op='sum', requires_grad=self._requires_grad)
        out = Ops.sum(self)
        out._prev = {self}
        out._requires_grad = self._requires_grad
        
        def _backward():
            if self._requires_grad:
                grad = np.ones_like(self._data) * out._grad
                self._grad = self._grad + grad if self._grad is not None else grad

        out._backward = _backward

        return out


    # def mean(self):
    #     out = Tensor(self._data.mean(), op='mean', requires_grad=self._requires_grad)
    #     out._prev = {self}
        
    #     def _backward():
    #         if self._requires_grad:
    #             grad = np.ones_like(self._data) / self._data.size * out._grad
    #             self._add_grad(grad)
    #     out._backward = _backward

    #     return out
    
    def __del__(self):
        if self._device == "cuda":
            if self.data is not None:
                # print("free gpu memory")
                Ops.free_gpu_memory(self.data)
            # if self.grad is not None:
            #     free_gpu_memory(self.grad.data)
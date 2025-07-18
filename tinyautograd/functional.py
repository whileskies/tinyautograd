from .tensor import Tensor
# from .rawtensor import Ops
import numpy as np
from .rawtensor import RawTensor

def relu(x: Tensor):
    out = Tensor(RawTensor.relu(x._data), op='relu', requires_grad=x._requires_grad)
    out._prev = {x}
    out._requires_grad = x._requires_grad

    def _backward():
        if x._requires_grad:
            grad = RawTensor.relu_grad(x._data) * out._grad
            x._add_grad(grad)
    out._backward = _backward
    return out


def log(x: Tensor):
    out = Tensor(RawTensor.log(x._data), op='log', requires_grad=x._requires_grad)
    out._prev = {x}

    def _backward():
        if x._requires_grad:
            grad = RawTensor.log_grad(x._data) * out._grad
            x._add_grad(grad)
    out._backward = _backward
    return out


def tanh(x: Tensor):
    t = RawTensor.tanh(x._data)
    out = Tensor(t, op='tanh', requires_grad=x._requires_grad)
    out._prev = {x}

    def _backward():
        if x._requires_grad:
            # grad = (1 - t ** 2) * out._grad
            grad = RawTensor.tanh_grad(t) * out._grad
            x._add_grad(grad)
    out._backward = _backward
    return out


def sigmod(x: Tensor):
    # s = 1 / (1 + np.exp(-x._data))
    s = RawTensor.sigmoid(x._data)
    out = Tensor(s, op='sigmod', requires_grad=x._requires_grad)
    out._prev = {x}

    def _backward():
        if x._requires_grad:
            # grad = s * (1 - s) * out._grad
            grad = RawTensor.sigmoid_grad(s) * out._grad
            x._add_grad(grad)
    out._backward = _backward
    return out


def softmax_cross_entropy(logits: Tensor, target: Tensor): # target 是 one-hot Tensor
    # e = np.exp(logits._data - np.max(logits._data, axis=-1, keepdims=True))
    # probs = e / e.sum(axis=-1, keepdims=True)
    probs = RawTensor.softmax(logits._data)
    # loss_data = -np.sum(target._data * np.log(probs + 1e-9), axis=-1, keepdims=True)
    loss = RawTensor.cross_entropy(probs, target._data)

    out = Tensor(loss, op='softmax_cross_entropy', requires_grad=logits._requires_grad)

    def _backward():
        if logits._requires_grad:
            # grad = probs - target._data
            grad = RawTensor.softmax_cross_entropy_grad(probs, target._data) * out._grad
            logits._add_grad(grad)

    out._backward = _backward
    out._prev = {logits, target}

    return out


def mse_loss(pred, target):
    return ((pred - target) ** 2).mean()
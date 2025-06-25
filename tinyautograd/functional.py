from .tensor import Tensor
import numpy as np

def relu(x: Tensor):
    out_data = np.maximum(0, x._data)
    out = Tensor(out_data, op='relu', requires_grad=x._requires_grad)
    out._prev = {x}

    def _backward():
        if x._requires_grad:
            grad = (x._data > 0).astype(np.float32) * out._grad
            x._add_grad(grad)
    out._backward = _backward
    return out

def log(x: Tensor):
    l = np.log(x._data)
    out = Tensor(l, op='log', requires_grad=x._requires_grad)
    out._prev = {x}

    def _backward():
        if x._requires_grad:
            grad = (1 / x._data) * out._grad
            x._add_grad(grad)
    out._backward = _backward
    return out


def tanh(x: Tensor):
    t = np.tanh(x._data)
    out = Tensor(t, op='tanh', requires_grad=x._requires_grad)
    out._prev = {x}

    def _backward():
        if x._requires_grad:
            grad = (1 - t ** 2) * out._grad
            x._add_grad(grad)
    out._backward = _backward
    return out


def sigmod(x: Tensor):
    s = 1 / (1 + np.exp(-x._data))
    out = Tensor(s, op='sigmod', requires_grad=x._requires_grad)
    out._prev = {x}

    def _backward():
        if x._requires_grad:
            grad = s * (1 - s) * out._grad
            x._add_grad(grad)
    out._backward = _backward
    return out


def softmax_cross_entropy(logits: Tensor, target: Tensor): # target 是 one-hot Tensor
    e = np.exp(logits._data - np.max(logits._data, axis=-1, keepdims=True))
    probs = e / e.sum(axis=-1, keepdims=True)
    loss_data = -np.sum(target._data * np.log(probs + 1e-9), axis=-1, keepdims=True)

    loss = Tensor(loss_data, op='softmax_cross_entropy', requires_grad=logits._requires_grad)

    def _backward():
        if logits._requires_grad:
            grad = probs - target._data
            logits._add_grad(grad)

    loss._backward = _backward
    loss._prev = {logits, target}
    return loss


def mse_loss(pred, target):
    # print('mse_loss:',((pred - target) ** 2).mean())
    return ((pred - target) ** 2).mean()
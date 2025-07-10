
import ctypes
import numpy as np


lib = ctypes.cdll.LoadLibrary("./libops.so")


lib.alloc_on_gpu.argtypes = [ctypes.c_int]
lib.alloc_on_gpu.restype = ctypes.c_void_p
lib.move_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.move_to_gpu.restype = ctypes.c_void_p
lib.move_to_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.free_gpu_memory.argtypes = [ctypes.c_void_p]


class Ops:
    def __init__(self):
        pass 

    
    @staticmethod
    def device(a, b):
        assert a.device == b.device, "Tensor a and b devices are not the same"
        return a.device
    
    @staticmethod
    def broadcast_shape(shape_a, shape_b):
        # 反向对齐
        result = []
        for a_dim, b_dim in zip(Ops._reverse(shape_a), Ops._reverse(shape_b)):
            if a_dim == b_dim:
                result.append(a_dim)
            elif a_dim == 1:
                result.append(b_dim)
            elif b_dim == 1:
                result.append(a_dim)
            else:
                raise ValueError(f"无法广播 shape: {shape_a} vs {shape_b}")
        
        # 加上多余的维度
        longer = shape_a if len(shape_a) > len(shape_b) else shape_b
        result += Ops._reverse(longer)[len(result):]

        return tuple(reversed(result))

    @staticmethod
    def _reverse(t):
        return t[::-1]
    
    @staticmethod 
    def tensor(a, b, data, op='', shape=None):
        from .tensor import Tensor
        if isinstance(b, Tensor):
            return Tensor(data, op, requires_grad=a.requires_grad or b.requires_grad, device=a.device, shape=shape)
        else:
            return Tensor(data, op, requires_grad=a.requires_grad, device=a.device, shape=shape)


    lib.add_vec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.add_vec_broadcast.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    @staticmethod
    def add(a, b):
        if Ops.device(a, b) == 'cuda':
            if a.size < b.size:
                a, b = b, a
            
            out = lib.alloc_on_gpu(a.size)

            if a.shape == b.shape:
                lib.add_vec(a.data, b.data, out, a.size)
            elif b.shape == (1, a.shape[1]):
                # 按行广播：(N, D) + (1, D)
                lib.add_vec_broadcast(a.data, b.data, out, a.size, a.shape[1], 0)
            elif b.shape == (a.shape[0], 1):
                # 按列广播：(N, D) + (N, 1)
                lib.add_vec_broadcast(a.data, b.data, out, a.size, a.shape[1], 1)

            elif b.shape == (1,):
                # 标量广播
                lib.add_vec_broadcast(a.data, b.data, out, a.size, 1, -1)
            else:
                raise ValueError(f"不支持的广播 shape: a.shape={a.shape}, b.shape={b.shape}")
            
            return Ops.tensor(a, b, out, op='+', shape=a.shape)
        else:
            c = a.data + b.data
            return Ops.tensor(a, b, c, op='+')


    lib.mul_vec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.mul_vec_broadcast.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]    
    @staticmethod
    def mul(a, b):
        if Ops.device(a, b) == 'cuda':
            if a.size < b.size:
                a, b = b, a
            
            out = lib.alloc_on_gpu(a.size)

            if a.shape == b.shape:
                lib.mul_vec(a.data, b.data, out, a.size)
            elif b.shape == (1, a.shape[1]):
                # 按行广播：(N, D) + (1, D)
                lib.mul_vec_broadcast(a.data, b.data, out, a.size, a.shape[1], 0)
            elif b.shape == (a.shape[0], 1):
                # 按列广播：(N, D) + (N, 1)
                lib.mul_vec_broadcast(a.data, b.data, out, a.size, a.shape[1], 1)

            elif b.shape == (1,):
                # 标量广播
                lib.mul_vec_broadcast(a.data, b.data, out, a.size, 1, -1)
            else:
                raise ValueError(f"不支持的广播 shape: a.shape={a.shape}, b.shape={b.shape}")
            
            return Ops.tensor(a, b, out, op='*', shape=a.shape)
        else:
            c = a.data * b.data
            return Ops.tensor(a, b, c, op='*')
    

    lib.launch_power.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    @staticmethod
    def pow(a, b):
        if a.device == 'cuda':
            fb = float(b)
            out = lib.alloc_on_gpu(a.size)
            lib.launch_power(a.data, out, fb, a.size)

            return Ops.tensor(a, b, out, op='**', shape=a.shape)
        else:
            c = a**b
            return Ops.tensor(a, b, c, op='**')

   
    lib.launch_power_grad.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    @staticmethod
    def pow_grad(a, b):
        if a.device == 'cuda':
            fb = float(b)
            out = lib.alloc_on_gpu(a.size)
            lib.launch_power_grad(a.data, out, fb, a.size)

            return Ops.tensor(a, b, out, op='**-grad', shape=a.shape)
        else:
            grad = b * (a.data ** (b - 1))
            return Ops.tensor(a, b, grad, op='**-grad')


    lib.launch_sum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.launch_sum_axis0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.launch_sum_axis1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    @staticmethod
    def sum(a, axis=None, keepdims=False):
        if a.device == 'cuda':    
            if axis is None:
                out = lib.alloc_on_gpu(1)
                lib.launch_sum(a.data, out, a.size)
                shape = a.shape if keepdims else (1,)

            elif axis == 0:
                N, M = a.shape
                out = lib.alloc_on_gpu(M)
                lib.launch_sum_axis0(a.data, out, N, M)
                shape = (1,M) if keepdims else (M,) 
                
            elif axis == 1:
                N, M = a.shape
                out = lib.alloc_on_gpu(N)
                lib.launch_sum_axis1(a.data, out, N, M)
                shape = (N, 1) if keepdims else (N,)
            else:
                raise ValueError(f"不支持的 axis: {axis}")
            
            return Ops.tensor(a, None, out, op='sum', shape=shape)
        else:
            d = a.data.sum(axis=axis, keepdims=keepdims)
            return Ops.tensor(a, None, d, op='sum')


    @staticmethod
    def to_device(a, device):
        if device == 'cuda':
            a_h = a.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            a_d = lib.move_to_gpu(a_h, a.data.size)
            return a_d
        else:
            n = a.size
            host_buffer = np.empty(n, dtype=np.float32)
            host_ptr = host_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            src_ptr = a.data
            lib.move_to_cpu(host_ptr, src_ptr, n)
            host_buffer = np.reshape(host_buffer, a.shape)

            return host_buffer
        

    @staticmethod
    def free_gpu_memory(data):
        lib.free_gpu_memory(data)
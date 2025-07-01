
import ctypes
import numpy as np


lib = ctypes.cdll.LoadLibrary("./libops.so")


lib.alloc_on_gpu.argtypes = [ctypes.c_int]
lib.alloc_on_gpu.restype = ctypes.c_void_p
lib.move_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.move_to_gpu.restype = ctypes.c_void_p
lib.move_to_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]

lib.add_vec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]


class Ops:
    def __init__(self):
        pass 

    @staticmethod
    def add(a , b):
        if a._device == 'cpu' and b._device == 'cpu':
            c = a.data + b.data
            return c, c.shape
        elif a.shape == b.shape:
            out = lib.alloc_on_gpu(a.size)
            lib.add_vec(a.data, b.data, out, a.size)
            return out, a.shape
        else:
            pass
        
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
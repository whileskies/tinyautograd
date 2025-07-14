
import ctypes
import numpy as np


lib = ctypes.cdll.LoadLibrary("./libops.so")


lib.alloc_on_gpu.argtypes = [ctypes.c_int]
lib.alloc_on_gpu.restype = ctypes.c_void_p
lib.move_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int]
lib.move_to_gpu.restype = ctypes.c_void_p
lib.move_to_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
lib.free_gpu_memory.argtypes = [ctypes.c_void_p]


class RawTensor:
    def __init__(self, data, device='cpu', shape=None):
        if device == 'cpu':
            if isinstance(data, (int, float)):
                self.data = np.array([data], dtype=np.float32)
            else:
                self.data = np.array(data, dtype=np.float32)
            self.shape = self.data.shape
            self.device = device
        elif device == 'cuda':
            # assert isinstance(data, ctypes.c_void_p)
            self.data = data
            self.shape = shape
            self.device = device
        else:
            raise  ValueError(f"Unsupported device: {device}")
    
    def __del__(self):
        if self.device == 'cuda':
            lib.free_gpu_memory(self.data)
        
    
    @staticmethod
    def device(a, b):
        assert a.device == b.device, "Tensor a and b devices are not the same"
        return a.device
    
    @property
    def size(self):
        return np.prod(self.shape)
    

    def to(self, device):
        if self.device == device:
            return
        if device == 'cuda':
            a_h = self.data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            a_d = lib.move_to_gpu(a_h, self.size)
            self.device = 'cuda'
            self.data = a_d
        elif device == 'cpu':
            n = self.size
            host_buffer = np.empty(n, dtype=np.float32)
            host_ptr = host_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            src_ptr = self.data
            lib.move_to_cpu(host_ptr, src_ptr, n)
            host_buffer = np.reshape(host_buffer, self.shape)
            self.device = 'cpu'
            self.data = host_buffer
        else:
            raise  ValueError(f"Unsupported device: {device}")
        return self

    
    # @staticmethod
    # def broadcast_shape(shape_a, shape_b):
    #     # 反向对齐
    #     result = []
    #     for a_dim, b_dim in zip(Ops._reverse(shape_a), Ops._reverse(shape_b)):
    #         if a_dim == b_dim:
    #             result.append(a_dim)
    #         elif a_dim == 1:
    #             result.append(b_dim)
    #         elif b_dim == 1:
    #             result.append(a_dim)
    #         else:
    #             raise ValueError(f"无法广播 shape: {shape_a} vs {shape_b}")
        
    #     # 加上多余的维度
    #     longer = shape_a if len(shape_a) > len(shape_b) else shape_b
    #     result += Ops._reverse(longer)[len(result):]

    #     return tuple(reversed(result))

    # @staticmethod
    # def _reverse(t):
    #     return t[::-1]
    

    lib.add_vec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.add_vec_broadcast.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    def __add__(self, other):
        if not isinstance(other, RawTensor):
            other = RawTensor(other, device='cpu')
            if self.device == 'cuda':
                other.to('cuda')
        
        a, b = self, other
        if RawTensor.device(a, b) == 'cuda':
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
            
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            r = a.data + b.data
            return RawTensor(r, device=a.device)
        
    def __radd__(self, other):
        return self + other


    lib.mul_vec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.mul_vec_broadcast.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]    
    def __mul__(self, other):
        if not isinstance(other, RawTensor):
            other = RawTensor(other, device='cpu')
            if self.device == 'cuda':
                other.to('cuda')

        a, b = self, other
        if RawTensor.device(a, b) == 'cuda':
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
            
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            r = a.data * b.data
            return RawTensor(r, device=a.device)
    

    def __rmul__(self, other):
        return self * other

    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)

    lib.launch_power.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supporting int/float powers for now"
        if self.device == 'cuda':
            fb = float(power)
            out = lib.alloc_on_gpu(self.size)
            lib.launch_power(self.data, out, fb, self.size)

            return RawTensor(out, device=self.device, shape=self.shape)
        else:
            r = self.data ** power
            return RawTensor(r, device=self.device)

   
    # lib.launch_power_grad.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    # @staticmethod
    # def pow_grad(a, b):
    #     if a.device == 'cuda':
    #         fb = float(b)
    #         out = lib.alloc_on_gpu(a.size)
    #         lib.launch_power_grad(a.data, out, fb, a.size)

    #         return Ops.tensor(out, device=a.device, op='**-grad', shape=a.shape)
    #     else:
    #         grad = b * (a.data ** (b - 1))
    #         return Ops.tensor(grad, device=a.device, op='**-grad')


    lib.launch_sum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.launch_sum_axis0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.launch_sum_axis1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    def sum(self, axis=None, keepdims=False):
        if self.device == 'cuda':    
            if axis is None:
                out = lib.alloc_on_gpu(1)
                lib.launch_sum(self.data, out, self.size)
                shape = (1,1) if keepdims else (1,)

            elif axis == 0:
                N, M = self.shape
                out = lib.alloc_on_gpu(M)
                lib.launch_sum_axis0(self.data, out, N, M)
                shape = (1,M) if keepdims else (M,) 
                
            elif axis == 1:
                N, M = self.shape
                out = lib.alloc_on_gpu(N)
                lib.launch_sum_axis1(self.data, out, N, M)
                shape = (N, 1) if keepdims else (N,)
            else:
                raise ValueError(f"不支持的 axis: {axis}")
            
            return RawTensor(out, device=self.device, shape=shape)
        else:
            d = self.data.sum(axis=axis, keepdims=keepdims)
            return RawTensor(d, device=self.device)


    lib.launch_fill.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    @staticmethod
    def ones_like(a):
        if a.device == 'cuda':
            sz = a.size
            out = lib.alloc_on_gpu(sz)
            lib.launch_fill(out, 1, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            d = np.ones_like(a)
            return RawTensor(d, device=a.device)


    # @staticmethod
    # def matmul(a, b):
    #     if Ops.device(a, b) == 'cuda':
    #         pass
    #     else:
    #         c = a.data @ b.data
    #         return Ops.tensor(c, device=a.device, op='matmul')
        
    
    # # Transpose
    # @staticmethod  
    # def T(a):
    #     if a.device == 'cuda':
    #         pass
    #     else:
    #         return Ops.tensor(a.T, device=a.device, op='T')
        

    # @staticmethod
    # def relu(a):
    #     if a.device == 'cuda':
    #         pass
    #     else:
    #         r = np.maximum(0, a.data)
    #         return Ops.tensor(r, device=a.device, op='relu')
        
    # @staticmethod
    # def log(a):
    #     if a.device == 'cuda':
    #         pass
    #     else:
    #         r = np.log(a.data)
    #         return Ops.tensor(r, device=a.device, op='log')
        

    # @staticmethod
    # def tanh(a):
    #     if a.device == 'cuda':
    #         pass
    #     else:
    #         r = np.tanh(a.data)
    #         return Ops.tensor(r, device=a.device, op='tanh')
        
    
    # @staticmethod
    # def sigmod(a):
    #     if a.device == 'cuda':
    #         pass
    #     else:
    #         r = 1 / (1 + np.exp(-x._data))
    #         return Ops.tensor(r, device=a.device, op='sigmod')
        


        
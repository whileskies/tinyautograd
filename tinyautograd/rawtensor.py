
import ctypes
import numpy as np


cuda = ctypes.cdll.LoadLibrary("./libops.so")


cuda.alloc_on_gpu.argtypes = [ctypes.c_int]
cuda.alloc_on_gpu.restype = ctypes.c_void_p
cuda.move_to_gpu.argtypes = [ctypes.c_void_p, ctypes.c_int]
cuda.move_to_gpu.restype = ctypes.c_void_p
cuda.move_to_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda.copy_to_cpu.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
cuda.free_gpu_memory.argtypes = [ctypes.c_void_p]


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
            # print('del')
            cuda.free_gpu_memory(self.data)
        
    
    @staticmethod
    def _device(a, b):
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
            a_d = cuda.move_to_gpu(a_h, self.size)
            self.device = 'cuda'
            self.data = a_d
        elif device == 'cpu':
            n = self.size
            host_buffer = np.empty(n, dtype=np.float32)
            host_ptr = host_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            src_ptr = self.data
            cuda.move_to_cpu(host_ptr, src_ptr, n)
            host_buffer = np.reshape(host_buffer, self.shape)
            self.device = 'cpu'
            self.data = host_buffer
        else:
            raise  ValueError(f"Unsupported device: {device}")
        return self


    @property
    def npdata(self):
        if self.device == 'cuda':
            n = self.size
            host_buffer = np.empty(n, dtype=np.float32)
            host_ptr = host_buffer.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            src_ptr = self.data
            cuda.copy_to_cpu(host_ptr, src_ptr, n)
            host_buffer = np.reshape(host_buffer, self.shape)
            return host_buffer
        else:
            return self.data
    

    cuda.add_vec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    cuda.add_vec_broadcast.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    def __add__(self, other):
        if not isinstance(other, RawTensor):
            other = RawTensor(other, device='cpu')
            if self.device == 'cuda':
                other.to('cuda')
        
        a, b = self, other
        if RawTensor._device(a, b) == 'cuda':
            if (len(a.shape), a.size) < (len(b.shape), b.size):
                a, b = b, a

            if len(a.shape) == 2 and len(b.shape) == 1 and b.shape[0] > 1:
                # 如果是行广播情况，如 a.shape = (N, D), b.shape = (D,)
                if len(a.shape) == 2 and b.shape[0] == a.shape[1]:
                    b.shape = (1, b.shape[0])  
                # 如果是列广播情况，如 a.shape = (N, D), b.shape = (N,)
                elif len(a.shape) == 2 and b.shape[0] == a.shape[0]:
                    b.shape = (b.shape[0], 1)
            
            out = cuda.alloc_on_gpu(a.size)

            if a.shape == b.shape:
                cuda.add_vec(a.data, b.data, out, a.size)
            elif len(a.shape) == 2 and len(b.shape) == 2 and b.shape[1] == a.shape[1]:
                # 按行广播：(N, D) + (1, D)
                cuda.add_vec_broadcast(a.data, b.data, out, a.size, a.shape[1], 0)
            elif len(a.shape) == 2 and len(b.shape) == 2 and b.shape[0] == a.shape[0]:
                # 按列广播：(N, D) + (N, 1)
                cuda.add_vec_broadcast(a.data, b.data, out, a.size, a.shape[1], 1)

            elif b.shape == (1,):
                # 标量广播
                cuda.add_vec_broadcast(a.data, b.data, out, a.size, 1, -1)
            else:
                raise ValueError(f"不支持的广播 shape: a.shape={a.shape}, b.shape={b.shape}")
            
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            r = a.data + b.data
            return RawTensor(r, device=a.device)
        
    def __radd__(self, other):
        return self + other


    cuda.mul_vec.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    cuda.mul_vec_broadcast.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]    
    def __mul__(self, other):
        if not isinstance(other, RawTensor):
            other = RawTensor(other, device='cpu')
            if self.device == 'cuda':
                other.to('cuda')

        a, b = self, other
        if RawTensor._device(a, b) == 'cuda':
            if (len(a.shape), a.size) < (len(b.shape), b.size):
                a, b = b, a

            if len(a.shape) == 2 and len(b.shape) == 1 and b.shape[0] > 1:
                # 如果是行广播情况，如 a.shape = (N, D), b.shape = (D,)
                if len(a.shape) == 2 and b.shape[0] == a.shape[1]:
                    b.shape = (1, b.shape[0])  
                # 如果是列广播情况，如 a.shape = (N, D), b.shape = (N,)
                elif len(a.shape) == 2 and b.shape[0] == a.shape[0]:
                    b.shape = (b.shape[0], 1)
            
            out = cuda.alloc_on_gpu(a.size)
            if a.shape == b.shape:
                cuda.mul_vec(a.data, b.data, out, a.size)
            elif len(a.shape) == 2 and len(b.shape) == 2 and b.shape[1] == a.shape[1]:
                # 按行广播：(N, D) + (1, D)
                cuda.mul_vec_broadcast(a.data, b.data, out, a.size, a.shape[1], 0)
            elif len(a.shape) == 2 and len(b.shape) == 2 and b.shape[0] == a.shape[0]:
                # 按列广播：(N, D) + (N, 1)
                cuda.mul_vec_broadcast(a.data, b.data, out, a.size, a.shape[1], 1)
            elif b.shape == (1,):
                # 标量广播
                cuda.mul_vec_broadcast(a.data, b.data, out, a.size, 1, -1)
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
    

    cuda.power.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    def __pow__(self, power):
        assert isinstance(power, (int, float)), "only supporting int/float powers for now"
        if self.device == 'cuda':
            fb = float(power)
            out = cuda.alloc_on_gpu(self.size)
            cuda.power(self.data, out, fb, self.size)

            return RawTensor(out, device=self.device, shape=self.shape)
        else:
            r = self.data ** power
            return RawTensor(r, device=self.device)


    def __truediv__(self, other):
        other = float(other)
    
        return self * other ** -1
    
    cuda.power_grad.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    @staticmethod
    def pow_grad(a, b):
        if a.device == 'cuda':
            fb = float(b)
            out = cuda.alloc_on_gpu(a.size)
            cuda.power_grad(a.data, out, fb, a.size)

            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            grad = b * (a.data ** (b - 1))
            return RawTensor(grad, device=a.device)


    cuda.sum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    cuda.sum_axis0.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    cuda.sum_axis1.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    def sum(self, axis=None, keepdims=False):
        if self.device == 'cuda':    
            if axis is None:
                out = cuda.alloc_on_gpu(1)
                cuda.sum(self.data, out, self.size)
                shape = (1,1) if keepdims else (1,)

            elif axis == 0:
                N, M = self.shape
                out = cuda.alloc_on_gpu(M)
                cuda.sum_axis0(self.data, out, N, M)
                shape = (1,M) if keepdims else (M,) 
                
            elif axis == 1:
                N, M = self.shape
                out = cuda.alloc_on_gpu(N)
                cuda.sum_axis1(self.data, out, N, M)
                shape = (N, 1) if keepdims else (N,)
            else:
                raise ValueError(f"不支持的 axis: {axis}")
            
            return RawTensor(out, device=self.device, shape=shape)
        else:
            d = self.data.sum(axis=axis, keepdims=keepdims)
            return RawTensor(d, device=self.device)

    
    def mean(self):
        if self.device == 'cuda':
            assert 1 == 0
        else:
            d = self.data.mean()
            return RawTensor(d, device=self.device)


    cuda.fill.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_int]
    @staticmethod
    def ones_like(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.fill(out, 1, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            d = np.ones_like(a.data)
            return RawTensor(d, device=a.device)

    cuda.matmul.argtypes = [
        ctypes.c_void_p,  # A
        ctypes.c_void_p,  # B
        ctypes.c_void_p,  # C
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    def __matmul__(self, other):
        assert isinstance(other, RawTensor), "Expected RawTensor operand"
        assert self.shape[-1] == other.shape[0], "Matrix shape mismatch"
        M, K = self.shape
        K2, N = other.shape
        # print(M, N, K)
        assert K == K2

        if RawTensor._device(self, other) == 'cuda':
            out = cuda.alloc_on_gpu(M * N)
            cuda.matmul(self.data, other.data, out, M, N, K)
            return RawTensor(out, device=self.device, shape=(M,N))

        else:
            out = np.matmul(self.data, other.data)
            return RawTensor(out, device=self.device)


    cuda.transpose_matrix.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int
    ]
    @property
    def T(self):
        if self.device == 'cuda':
            assert len(self.shape) == 2
            M, N = self.shape
            out = cuda.alloc_on_gpu(M * N)
            cuda.transpose_matrix(self.data, out, M, N)
            return RawTensor(out, device=self.device, shape=(N,M))
        else:
            return RawTensor(self.data.T, device=self.device)


    cuda.launch_relu.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    @staticmethod
    def relu(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_relu(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            r = np.maximum(0, a.data)
            return RawTensor(r, device=a.device)
        

    cuda.launch_relu_grad.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    @staticmethod
    def relu_grad(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_relu_grad(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            grad = (a.data > 0).astype(np.float32)
            return RawTensor(grad, device=a.device)
        

    cuda.launch_log.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    @staticmethod
    def log(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_log(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            r = np.log(a.data)
            return RawTensor(r, device=a.device)

    cuda.launch_log_grad.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    @staticmethod
    def log_grad(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_log_grad(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            grad = (1 / a.data)
            return RawTensor(grad, device=a.device)    


    cuda.launch_tanh.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    @staticmethod
    def tanh(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_tanh(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            r = np.tanh(a.data)
            return RawTensor(r, device=a.device)

    cuda.launch_tanh_grad.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    @staticmethod
    def tanh_grad(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_tanh_grad(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            t = a.data  # a = np.tanh(x)
            grad = (1 - t ** 2)  
            return RawTensor(grad, device=a.device) 


    cuda.launch_sigmoid.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    @staticmethod
    def sigmoid(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_sigmoid(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            r = 1.0 / (1.0 + np.exp(-a.data))
            return RawTensor(r, device=a.device)


    cuda.launch_sigmoid_grad.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int
    ]

    @staticmethod
    def sigmoid_grad(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_sigmoid_grad(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            s = a.data  # a = sigmoid(x)
            grad = s * (1.0 - s)
            return RawTensor(grad, device=a.device)  
        
    cuda.launch_exp.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int
    ]
    @staticmethod
    def exp(a):
        if a.device == 'cuda':
            sz = a.size
            out = cuda.alloc_on_gpu(sz)
            cuda.launch_exp(a.data, out, sz)
            return RawTensor(out, device=a.device, shape=a.shape)
        else:
            r = np.exp(a.data)
            return RawTensor(r, device=a.device)
        
    
    cuda.launch_softmax_simple.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    @staticmethod
    def softmax(logits):
        if logits.device == 'cuda':
            B, C = logits.shape
            out = cuda.alloc_on_gpu(B * C)
            cuda.launch_softmax_simple(logits.data, out, B, C)
            return RawTensor(out, shape=(B, C), device='cuda')
        else:
            e = np.exp(logits.data - np.max(logits.data, axis=-1, keepdims=True))
            probs = e / e.sum(axis=-1, keepdims=True)
            # loss_data = -np.sum(target._data * np.log(probs + 1e-9), axis=-1, keepdims=True)
            return RawTensor(probs, device=logits.device)


    cuda.launch_cross_entropy_loss_simple.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int
    ]
    @staticmethod
    def cross_entropy(probs, target):
        if RawTensor._device(probs, target) == 'cuda':
            B, C = probs.shape
            out = cuda.alloc_on_gpu(B)
            cuda.launch_cross_entropy_loss_simple(probs.data, target.data, out, B, C)
            return RawTensor(out, device=probs.device, shape=(B,))
        else:
            loss = -np.sum(target.data * np.log(probs.data + 1e-9), axis=-1, keepdims=True)
            return RawTensor(loss, device=probs.device)



    cuda.launch_softmax_cross_entropy_grad.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int
    ]
    @staticmethod
    def softmax_cross_entropy_grad(probs, target):
        assert probs.shape == target.shape
        if RawTensor._device(probs, target) == 'cuda':
            B, C = probs.shape
            grad = cuda.alloc_on_gpu(B * C)
            cuda.launch_softmax_cross_entropy_grad(
                probs.data, target.data, grad, B, C
            )
            return RawTensor(grad, device=probs.device, shape=probs.shape)  
        else:
            grad = probs.data - target.data
            return RawTensor(grad, device=probs.device)

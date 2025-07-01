from .ops import *
import numpy as np
from .tensor import *
import time

def test_add_vec():
    a = [[1, 1, 1], [1, 1, 1]]
    b = [[2, 2, 2], [2, 2, 2]]
    c = [[3, 3, 3], [3, 3, 3]]
    ta = Tensor(a)
    tb = Tensor(b)
    tc = Tensor(c)
    ta.to('cuda')
    tb.to('cuda')
    tc.to('cuda')

    res = Tensor(Ops.add(ta, tb), device='cuda')
    res = Tensor(Ops.add(res, tc), device='cuda')

    res.to('cpu')
    print(res.data)
    print(res.shape)

    
def test_add_vec_large():
    N = 50000000  
    a = np.ones((3, N), dtype=np.float32)
    b = np.full((3, N), 2.0, dtype=np.float32)
    c = np.full((3, N), 3.0, dtype=np.float32)
    print(a.shape)

    # ===== CPU 计算 =====
    ta = Tensor(a)
    tb = Tensor(b)
    tc = Tensor(c)

    start_cpu = time.time()
    rs_cpu = Tensor(Ops.add(ta, tb))
    res_cpu = Tensor(Ops.add(rs_cpu, tc))
    cpu_time = time.time() - start_cpu

    res_cpu_np = res_cpu.data  # NumPy 结果

    # ===== CUDA 计算 =====
    ta_gpu = ta.to('cuda')
    tb_gpu = tb.to('cuda')
    tc_gpu = tc.to('cuda')

    start_gpu = time.time()
    res_gpu = Tensor(Ops.add(ta_gpu, tb_gpu), device='cuda')
    res_gpu = Tensor(Ops.add(res_gpu, tc_gpu), device='cuda')
    gpu_time = time.time() - start_gpu

    res_gpu_cpu = res_gpu.to('cpu')
    res_gpu_np = res_gpu_cpu.data

    # ===== 比较结果与时间 =====
    print(f"CPU time:  {cpu_time * 1000:.2f} ms")
    print(f"CUDA time: {gpu_time * 1000:.2f} ms")
    print(f"Max error: {np.abs(res_gpu_np - res_cpu_np).max()}")
    print("First 3 results:", res_gpu_np[:3])
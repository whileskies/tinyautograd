from .rawtensor import *
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

    res = Ops.add(ta, tb)
    res = Ops.add(res, tc)

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
    rs_cpu = Ops.add(ta, tb)
    res_cpu = Ops.add(rs_cpu, tc)
    cpu_time = time.time() - start_cpu

    res_cpu_np = res_cpu.data  # NumPy 结果

    # ===== CUDA 计算 =====
    ta_gpu = ta.to('cuda')
    tb_gpu = tb.to('cuda')
    tc_gpu = tc.to('cuda')

    start_gpu = time.time()
    res_gpu = Ops.add(ta_gpu, tb_gpu)
    res_gpu = Ops.add(res_gpu, tc_gpu)
    gpu_time = time.time() - start_gpu

    res_gpu_cpu = res_gpu.to('cpu')
    res_gpu_np = res_gpu_cpu.data

    # ===== 比较结果与时间 =====
    print(f"CPU time:  {cpu_time * 1000:.2f} ms")
    print(f"CUDA time: {gpu_time * 1000:.2f} ms")
    print(f"Max error: {np.abs(res_gpu_np - res_cpu_np).max()}")
    print("First 3 results:", res_gpu_np[:3])


def test_vec_add_broadcast():
    a = [[1, 1, 1], [3, 3, 3]]
    b = [[2, 2, 2]]
    ta = Tensor(a)
    tb = Tensor(b)
    ta.to('cuda')
    tb.to('cuda')

    res = Ops.add(ta, tb)

    res.to('cpu')
    print(res.data)
    print(res.shape)


def test_vec_mul():
    a = [[1, 1, 1], [3, 3, 3]]
    b = [5]
    # a,b = b,a
    ta = Tensor(a)
    tb = Tensor(b)
    ta.to('cuda')
    tb.to('cuda')

    res = Ops.mul(ta, tb)

    res.to('cpu')
    print(res.data)
    print(res.shape)


def test_vec_mul_broadcast():
    a = [[1, 1, 1], [3, 3, 3]]
    b = [[2, 2, 2]]
    ta = Tensor(a)
    tb = Tensor(b)
    ta.to('cuda')
    tb.to('cuda')

    res = Ops.mul(ta, tb)

    res.to('cpu')
    print(res.data)
    print(res.shape)


def test_power_large():
    N = 100_000_000  
    x = np.full((3, N), 2.0, dtype=np.float32)
    power = 3.0

    print("Input shape:", x.shape)

    # ===== CPU 计算 =====
    tx = Tensor(x)
    start_cpu = time.time()
    res_cpu = tx ** power  # 调用 Tensor.__pow__
    cpu_time = time.time() - start_cpu
    res_cpu_np = res_cpu.data  # NumPy 结果

    # ===== CUDA 计算 =====
    tx_gpu = tx.to('cuda')
    start_gpu = time.time()
    res_gpu = Ops.pow(tx_gpu, power)
    gpu_time = time.time() - start_gpu

    res_gpu_cpu = res_gpu.to('cpu')
    res_gpu_np = res_gpu_cpu.data

    # ===== 比较结果与时间 =====
    print(f"CPU time:  {cpu_time * 1000:.2f} ms")
    print(f"CUDA time: {gpu_time * 1000:.2f} ms")
    print(f"Max error: {np.abs(res_gpu_np - res_cpu_np).max()}")
    print("First 3 values:", res_gpu_np.ravel()[:3])


def test_sum():
    for i in range (10000, 20000):
        a = [1] * i
        # print(i, len(a))
        ta = Tensor(a)
        ta.to('cuda')
        tb = Ops.sum(ta)
        tb.to('cpu')
        assert tb.data[0] == len(a)

def test_sum_axis0():
    a0 = [1] * 12345
    a1 = [2] * 12345
    a = [a0, a1]
    na = np.array(a)
    nas = na.sum(axis=0, keepdims=False)
    print(nas[0], nas.shape)
    ta = Tensor(a)
    ta.to('cuda')
    tb = Ops.sum(ta, axis=0, keepdims=False)
    tb.to('cpu')
    print(tb.data[0], tb.shape)

def test_sum_axis1():
    for i in range (10000, 12345):
        # print(i)
        a0 = [1] * i
        a1 = [2] * i
        a2 = [3] * i
        a = [a0, a1, a2]
        na = np.array(a)
        nas = na.sum(axis=1, keepdims=True)
        # print(nas, nas.shape)
        ta = Tensor(a)
        ta.to('cuda')
        tb = Ops.sum(ta, axis=1, keepdims=True)
        tb.to('cpu')
        # print(tb.data, tb.shape)
        assert nas[0] == tb.data[0] and nas[1] == tb.data[1] and nas[2] == tb.data[2]


def test_ones_like():
    a = [[1, 2, 3], [4, 5, 6]]
    ta = Tensor(a)
    ta.to('cuda')
    tb = Ops.ones_like(ta)
    tb.to('cpu')
    print(tb.data)
    print(tb.shape)

    na = np.array(a)
    nb = np.ones_like(na)
    print(nb)
    print(nb.shape)
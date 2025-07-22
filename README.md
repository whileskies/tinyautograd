# tinyautograd
TinyAutoGrad is an automatic differentiation engine and neural network library inspired by [Micrograd](https://github.com/karpathy/micrograd), with CUDA support.

# cpu
python -m samples.mnist.mnist 

# cuda 🚧
nvcc -shared -o libops.so tinyautograd/ops.cu  -Xcompiler -fPIC -lcublas

python -m samples.mnist.mnist_cuda

pytest tinyautograd/test_rawtensor.py -s -v --cache-clear
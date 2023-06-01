# JAX-vs-PyTorch-on-GPU

This project aims to compare between the computing speed of JAX and PyTorch on a various models of NVIDIA GPUs.
I aim to compare the following typical use cases:

1. Training a neural network on a single GPU.
2. Training a neural network on multiple GPUs.
3. Computing per-sample gradients of a neural network on a single GPU.
4. Computing per-sample gradients of a neural network on multiple GPUs.

Limited by my access to GPUs, I will only be able to test on the following GPUs:

1. NVIDIA GTX 2080 Ti
2. NVIDIA GTX 3080
3. NVIDIA A100

The benchmarks I will be using are:

1. MNIST
2. CIFAR-10
3. NLP (TBD)

The results will be updated in an open Wandb project and this README file in ``real-time''.

## Results


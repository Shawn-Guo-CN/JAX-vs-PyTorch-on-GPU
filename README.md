# JAX-vs-PyTorch-on-GPU

This project aims to compare between the computing speed of JAX and PyTorch on a various models of NVIDIA GPUs.
We aim to compare the following typical use cases:

1. Training a neural network on a single GPU.
2. Training a neural network on multiple GPUs.
3. Computing per-sample gradients of a neural network on a single GPU.
4. Computing per-sample gradients of a neural network on multiple GPUs.

Limited by ours access to GPUs, we will only be able to test on the following GPUs:

1. NVIDIA GTX 2080 Ti
2. NVIDIA GTX 3080
3. NVIDIA A5000
4. NVIDIA A100

The benchmarks we will be using are:

1. MNIST
2. CIFAR-10
3. NLP (TBD)
4. GNN (TBD)

The results will be updated in an open Wandb project and this README file in ``real-time''.

## Pipeline

### 1. Main Python Script

We assume the clusters running the comparison experiments use Slurm as the job scheduler.
To run a single comparison (on a specific GPU, benchmark, and use case), we will first call the `main.py` script with the following arguments:

```bash
python main.py --benchmark <benchmark> --use_case <use_case_number>
```

Note that the `use_case` argument corresponds to the use case number in the list above, e.g. `1` for the task of training a neural network on a single GPU.

More importantly, the GPU model is not specified in the above command, since we rely on `torch.cuda.get_device_name()` to automatically detect and log the GPU model.

### 2. Slurm Scripts

By running the `main.py` script, the programme will automatically generate **TWO** Slurm scripts in the `slurm_scripts` folder, one for running the JAX experiment and the other for running the PyTorch experiment.

The Slurm scripts will be named as follows:

```bash
slurm_scripts/<framework>_<benchmark>_<use_case_number>.sh
```

The `main.py` function will then submit the Slurm scripts to the cluster.

### 3. `run.py` in Slurm Jobs

In the Slurm job for running JAX or PyTorch, we will first call the `run.py` script with the following arguments:

```bash
python <framework>/run.py --benchmark <benchmark> --use_case <use_case_number>
```


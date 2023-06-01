# JAX-vs-PyTorch-on-GPU

This project aims to compare between the computing speed of JAX and PyTorch on a various models of NVIDIA GPUs.
We aim to compare the following typical use cases:

1. Training a neural network on a given benchmark.
3. Computing per-sample gradients of a neural network on a given benchmark.

For each use case, we will compare the computing speed of JAX and PyTorch on the following benchmarks:

1. MNIST
2. CIFAR-10
3. NLP (TBD)
4. GNN (TBD)

At the same time, for each use case and benchmark, we will compare the computing speed of JAX and PyTorch on a single GPU, and on multiple GPUs, by setting the `multi_gpu` flag in the `main.py` script (as stated below).

Limited by ours access to GPUs, we will only be able to test on the following GPUs:

1. NVIDIA GTX 2080 Ti
2. NVIDIA GTX 3080
3. NVIDIA A5000
4. NVIDIA A100

The results will be updated in an open Wandb project and this README file in ``real-time''.

## Pipeline

### 1. Main Python Script

We assume the clusters running the comparison experiments use Slurm as the job scheduler.
To run a single comparison (on a specific GPU, benchmark, and use case), we will first call the `main.py` script with the following arguments:

```bash
python main.py --benchmark <benchmark> --use_case <use_case_number> <--multi_gpu>
```

Note that the `use_case` argument corresponds to the use case number in the list above, e.g. `1` for the task of training a neural network on a single GPU.

The `multi_gpu` is used to control whether we want to run the experiment on a single GPU or all GPUs available.

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
python <framework>/run.py --benchmark <benchmark> --use_case <use_case_number> <--multi_gpu>
```


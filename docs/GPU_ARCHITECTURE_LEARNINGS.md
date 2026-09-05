# GPU Architecture Learnings

Findings from implementing and benchmarking Metal GPU support for Neuron, using MNIST training
as the workload.

> **Status:** the implementation these findings came from is **no longer in the codebase**. The
> encoder-holder architecture (`MetalEncoderHolder`, `MetalCommandEncoder`, `syncAndReplace` /
> `syncAndFinish`, `packToNCHW` / `unpackFromNCHW`, the `neuron_instance_norm` and
> `neuron_max_pool_2x2` kernels) was removed. What survives in `Sources/Neuron/Devices/GPU/` is
> `MetalTensorStorage`, `MetalContext`, `BufferPool`, and `Resources/GPU.metal`. This document is
> kept for the measurements and the root-cause analysis, which still apply to any future attempt.

## Performance Summary

- **CPU MNIST (batch 64):** ~0.12s per batch
- **GPU MNIST (batch 64):** ~0.5–0.6s per batch
- **Result:** GPU was ~5× slower than CPU for this workload

## What We Tried

### 1. Metal InstanceNorm

- **Goal:** Eliminate the sync before InstanceNorm (it ran on CPU and had to read ReLu output).
- **Result:** No performance change.
- **Implementation:** Per-`(n, c)` normalization over the spatial dims.

### 2. Metal MaxPool

- **Goal:** Extend the GPU segment to include MaxPool, moving the sync later in the pass.
- **Result:** No performance change.
- **Implementation:** 2×2 pooling kernel with an indices buffer for the backward pass.

### 3. Multiple GPU Workers

- **Goal:** Match CPU parallelism (4–8 workers) with per-worker Metal encoders instead of one
  shared encoder.
- **Result:** **Worse** (~0.7s). More command buffers and more syncs increased overhead.

### 4. Single Worker

- **Conclusion:** A single worker processing the full batch beats multiple workers on smaller
  chunks. GPU parallelism does not decompose the way CPU worker parallelism does.

## Root Cause Analysis

Why the GPU path lost:

1. **Sync overhead** — two `waitUntilCompleted()` calls per batch block the CPU, each adding
   latency. The pipeline needed them wherever a CPU layer had to read a Metal-backed tensor.
2. **Small workload** — MNIST at 28×28, batch 64, has limited compute. Kernel launch and sync
   overhead dominate the actual GPU work.
3. **Backward pass on CPU** — gradients were computed on CPU. If the Conv2d/ReLu/Dense backward
   paths are CPU-only, most of training time is in backward, where the GPU never participates.
4. **Pack/unpack cost** — every batched layer copied data into and out of Metal buffers. For
   small tensors that copy is a significant fraction of the work.
5. **CPU parallelism** — the CPU path uses `Constants.maxWorkers` (typically 4) across batch
   chunks. The GPU path used one worker, and adding more made it worse (see above).

## Recommendations For A Future Attempt

1. **Profile first** — Instruments (Time Profiler, Metal System Trace) before optimizing.
2. **Larger workloads** — CIFAR-10, larger images, or bigger models. GPU may win once compute
   dominates the fixed overhead.
3. **Metal backward** — implement Conv2d/ReLu/Dense backward on Metal so gradients stay on the
   GPU. Without this the GPU sits idle for most of each step.
4. **Async execution** — `addCompletedHandler` instead of `waitUntilCompleted`, to overlap CPU
   and GPU work. Requires a training-loop refactor.
5. **Accept the trade-off** — for MNIST-sized workloads the CPU may simply remain faster. GPU
   support earns its keep on larger models and on deployment targets where it matters.

## Metal Shading Language Notes

These are language-level and remain true regardless of the architecture above:

- **No lambdas** — use inline logic or helper functions. Lambdas produce
  "lambda expressions are not supported in Metal" errors.
- **NCHW layout** — batched tensors index as `n*C*H*W + c*H*W + h*W + w`.
- **Indices buffers** — for MaxPool backward, store a `uint32` (0–3) per output element. A
  float-typed storage buffer can be reused and reinterpreted, since the byte size matches.

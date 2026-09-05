# GPU-accelerated neural networks with Metal in Swift

**The single most impactful change for your library is backing the Tensor class directly with `MTLBuffer` instead of `ContiguousArray<Float>`.** On Apple Silicon's unified memory, a shared-mode `MTLBuffer` gives both CPU and GPU zero-copy access to the same physical memory — meaning every operation that currently copies data into a temporary buffer before GPU dispatch is doing unnecessary work. This, combined with encoding all layer operations into a single `MTLCommandBuffer` (eliminating CPU-GPU round-trips between layers), should transform GPU performance from "no improvement" to significant speedup for any tensor larger than roughly 128×128 elements. The architecture lessons from Apple's own MLX framework — lazy evaluation, buffer pooling, and command buffer batching — provide a proven blueprint for building this efficiently in Swift.

---

## The Tensor class must own an MTLBuffer, not a ContiguousArray

The previous GPU implementation likely showed no improvement because of excessive data copying. When tensors use `ContiguousArray<Float>` as primary storage, every GPU dispatch requires calling `makeBuffer(bytes:length:options:)`, which **physically copies data** into a new Metal buffer. On Apple Silicon, this copy is entirely unnecessary.

The fix is architectural: **`MTLBuffer` with `.storageModeShared` should be the Tensor's primary storage**. On Apple Silicon, CPU and GPU share the same physical DRAM. A shared-mode buffer gives you a CPU-accessible pointer via `buffer.contents()` and is simultaneously usable by any Metal compute encoder — no copies, no transfers, no blit operations. The recommended Tensor structure looks like this:

```swift
class Tensor {
    let buffer: MTLBuffer
    let size: TensorSize  // rows, columns, depth

    var cpuPointer: UnsafeMutablePointer<Float> {
        buffer.contents().assumingMemoryBound(to: Float.self)
    }

    init(device: MTLDevice, size: TensorSize) {
        let bytes = size.elementCount * MemoryLayout<Float>.stride
        self.buffer = device.makeBuffer(length: bytes, options: .storageModeShared)!
        self.size = size
    }
}
```

This is exactly what MLX does internally — arrays live in unified memory and are accessible from any device without transfer. The `cpuPointer` property lets you use Accelerate/vDSP on the same memory for CPU-side operations, while `encoder.setBuffer(tensor.buffer, offset: 0, index: N)` passes it directly to GPU compute kernels. For truly tiny transient data under **4 KB**, Metal's `setBytes(_:length:index:)` on the encoder avoids even needing a buffer object.

One critical caveat: **cache coherency is only guaranteed at command buffer boundaries**. After CPU writes to a shared buffer, the command buffer that reads it must be committed *after* writing completes. After GPU writes, you must wait for command buffer completion before reading on CPU. Violating this ordering causes silent data corruption.

---

## Encode the entire forward and backward pass in one command buffer

The second major performance killer is creating and committing separate command buffers for each layer. Every `commandBuffer.commit()` + `waitUntilCompleted()` round-trip costs **20–50 microseconds** on macOS (approximately 5× more on iOS). For a 10-layer network, that's 200–500µs of pure overhead per pass — potentially exceeding the actual computation time for small models.

The correct pattern encodes all layers into a **single command buffer with sequential compute dispatches**. Metal guarantees that within a single compute command encoder using default serial dispatch mode, all memory writes from one dispatch are visible to subsequent dispatches. This means a conv→batchnorm→relu chain works correctly without explicit barriers or synchronization:

```swift
let cmdBuffer = commandQueue.makeCommandBuffer()!
let encoder = cmdBuffer.makeComputeCommandEncoder()!

for layer in network.layers {
    encoder.setComputePipelineState(layer.pipeline)
    encoder.setBuffer(layer.input.buffer, offset: 0, index: 0)
    encoder.setBuffer(layer.output.buffer, offset: 0, index: 1)
    encoder.setBuffer(layer.weights.buffer, offset: 0, index: 2)
    encoder.dispatchThreads(layer.gridSize, threadsPerThreadgroup: layer.threadgroupSize)
}

encoder.endEncoding()
cmdBuffer.commit()
// Only waitUntilCompleted() at batch boundary, not between layers
```

The backward pass and weight update can go into the same command buffer. **Pre-allocate all activation, gradient, and weight buffers at network initialization** — never allocate during training loops, as `makeBuffer` involves kernel calls and zero-filling. MLX takes this further by maintaining a **buffer cache/pool** that recycles freed `MTLBuffer` objects for future allocations of similar sizes, avoiding repeated allocation overhead. Implementing a simple size-bucketed buffer pool (rounding up to powers of 2) provides most of this benefit.

For training pipelines, Apple's Metal Best Practices Guide recommends **triple buffering**: maintain 3 copies of per-batch input buffers, use a `DispatchSemaphore(value: 3)`, and let the CPU prepare batch N+1 while the GPU processes batch N. This keeps both processors busy and eliminates idle time.

---

## Batch dispatch strategy: pack everything into one kernel call

Both MLX and MPS pack the entire batch into a single contiguous buffer and process it with **one kernel dispatch**. The kernel grid is sized to cover `batch_size × spatial_dimensions`, with each thread computing its batch and spatial index from its global thread position. MLX's convolution kernels demonstrate this pattern:

```metal
int n = (gid.z) / out_pixels;  // batch index
int oS = (gid.z) % out_pixels; // spatial index
```

Per-sample dispatches should be avoided entirely. The overhead of encoding and dispatching a kernel (~5–50µs per dispatch) makes per-sample processing catastrophically slow for batched training. A single dispatch with a grid covering all samples amortizes this overhead completely. Apple's WWDC guidance is explicit: "Submitting work in larger volumes is an easy way for your application to scale and reach its potential."

MLX further optimizes by **reusing the active `MTLComputeCommandEncoder`** across multiple operations. Rather than creating a new encoder for each operation, `device.get_command_encoder(stream_index)` returns the currently active encoder. Multiple compute pipelines are set and dispatched on the same encoder until a configurable limit is hit or synchronization forces a flush.

---

## MPSGraph is the strongest option for standard operations

For the core neural network operations — convolution, transposed convolution, matrix multiplication, and their gradients — **MPSGraph provides the best performance-to-engineering-effort ratio** by a wide margin. It wraps Apple's hand-tuned MPS kernels with graph-level optimizations that are extremely difficult to replicate.

**Automatic differentiation** is built in. Calling `graph.gradients(of: lossTensor, with: [weights, biases])` automatically constructs the backward pass using chain rule propagation, with dead code elimination and constant folding applied to the gradient graph. For convolution, MPSGraph provides `convolution2DDataGradient` and `convolution2DWeightsGradient` — these call into the same optimized kernels as `MPSCNNConvolutionGradient`, which would take enormous effort to match with custom kernels.

The most powerful MPSGraph feature for performance is **operation stitching**. The Metal compiler fuses adjacent elementwise operations into the preceding hand-tuned kernel. For example, a conv→bias→GeLU chain gets compiled into a single optimized shader. Apple reports this makes GeLU **10–50× faster** than separate kernel dispatches. This fusion happens automatically for "stitchable" operations adjacent to convolution, matrix multiplication, or reduction kernels.

MPSGraph is available on **macOS 11+, iOS 14+, and visionOS**, covering all target platforms. It can target GPU, CPU, and Neural Engine. For standalone matrix multiplication outside of graph context, `MPSMatrixMultiplication` benchmarks show custom Metal kernels typically achieve only **~50% of MPS performance** on M1 Max. The exception is MLX's heavily optimized matmul using `simdgroup_matrix` operations — but that represents months of per-chip tuning.

| Operation | Recommended approach | Fallback |
|---|---|---|
| 2D convolution (forward + backward) | MPSGraph `convolution2D` + auto-diff | MPS `MPSCNNConvolution` + `MPSCNNConvolutionGradient` |
| Transposed convolution | MPSGraph `convolutionTranspose2D` | Custom im2col + GEMM kernel |
| Matrix multiplication | MPSGraph `matrixMultiplication` or `MPSMatrixMultiplication` | Custom kernel with `simdgroup_matrix` |
| Elementwise (ReLU, sigmoid) | MPSGraph stitching (fused into adjacent ops) | Simple custom kernel (memory-bound, easy to write) |
| Batch/layer normalization | MPSGraph `batchNormalization` | Custom reduction + elementwise kernel |

---

## What MLX's architecture teaches about device abstraction

MLX's design makes several unconventional choices worth emulating. **Arrays are not bound to a device** — instead, *operations* specify where they run. An array in MLX lives in unified memory and can be consumed by either CPU or GPU operations without transfer. This aligns perfectly with Apple Silicon's architecture and simplifies the API dramatically compared to PyTorch's `.to(device)` pattern.

MLX's **lazy evaluation model** is its most important optimization. Operations like `mx.add(a, b)` create graph nodes but perform no computation. Only when `mx.eval()` is called does the framework traverse the graph, batch operations into command buffers, and dispatch. This enables three key optimizations: operation fusion (combining multiple elementwise ops into single kernels), dead code elimination (never computing unused intermediates), and optimal memory reuse (buffers from completed operations are immediately recycled). The `mx.compile()` API goes further by JIT-compiling entire subgraphs into optimized Metal kernels.

MLX's **MetalAllocator** implements a buffer cache with configurable limits. When arrays are freed, their underlying `MTLBuffer` objects are kept in a pool keyed by size for reuse. The cache limit defaults to `1.5× device.recommendedMaxWorkingSetSize`. This is essential for training loops where the same buffer sizes are needed every iteration. A simpler version for a custom library would use size-bucketed pools with LRU eviction.

For cross-device dependencies, MLX uses **`MTLFence` objects** to synchronize between streams without expensive full barriers. When a GPU operation depends on a CPU result (or vice versa), fences provide fine-grained ordering guarantees within a command queue.

---

## When GPU dispatch actually pays off vs Accelerate

The crossover point where Metal GPU compute beats CPU (via Accelerate/vDSP/BLAS with AMX coprocessor) varies by operation type and chip generation. Benchmark data from multiple sources converges on these practical thresholds:

**Matrix multiplication** shows the clearest story. On M1, GPU achieves **1.36 TFLOPS** vs CPU's **0.90 TFLOPS** — a 1.5× advantage. On M4, the gap widens to **2.90 vs 1.49 TFLOPS** (1.95×). But this advantage only materializes for matrices **larger than approximately 128×128** (~16K elements). Below this size, the ~20-50µs dispatch overhead dominates. For the dense layers in a typical neural network, this means batched operations (batch_size × hidden_size) should almost always go to GPU, while per-sample operations on small hidden dimensions may be faster on CPU.

**Convolution** almost universally benefits from GPU above **32×32 spatial dimensions with 32+ channels**. The arithmetic intensity of convolution (many multiplies per memory access) makes it compute-bound, which favors the GPU's parallel architecture. Even small convolutions in neural networks typically exceed the crossover point.

**Elementwise operations** (activations, normalization) are memory-bound (~0.08 FLOP/byte for ReLU). The GPU wins above roughly **10K–100K elements**, but the real win comes from **fusing** these into adjacent compute-bound kernels (conv + relu as one dispatch) rather than dispatching them separately.

**Critical insight**: the question isn't just "is this single operation faster on GPU?" but "is the entire layer pipeline faster when everything stays on GPU?" Even if individual small operations would be faster on CPU, **the cost of switching between CPU and GPU execution** (synchronizing command buffers, potential cache invalidation) often exceeds the per-operation savings. For training, the recommended strategy is: run everything above a minimum threshold on GPU within a single command buffer, and only fall back to CPU (Accelerate) for operations genuinely too small to benefit — typically tensors with fewer than ~1,000 elements.

---

## Custom Metal kernel design for when you need it

For operations not covered by MPS/MPSGraph or where fusion requires custom logic, understanding Apple Silicon's GPU architecture is essential. All Apple Silicon GPUs use a **SIMD group width of 32 threads**. Threadgroup sizes should be multiples of 32, with **256 threads** (8 SIMD groups) being the practical sweet spot for most ML kernels. The maximum is 1024, but complex kernels with high register pressure may be limited to 512 or less — always query `pipeline.maxTotalThreadsPerThreadgroup` at runtime.

For convolution, the **implicit GEMM approach** (computing im2col indices on-the-fly within the matrix multiply kernel) avoids the massive memory overhead of explicit im2col while matching its computational pattern. Apple's MPS uses this internally. For 3×3 filters specifically, **Winograd-domain convolution** with `simdgroup_matrix` operations (available on Apple7+ / M1+) can significantly reduce arithmetic complexity — MLX implements this in its `conv.metal` kernels.

The `simdgroup_matrix<T, 8, 8>` type and `simdgroup_multiply_accumulate(D, A, B, C)` intrinsic are Apple's equivalent of NVIDIA tensor cores. Each SIMD group processes **8×8 matrix tiles**, performing 512 multiplications in ~20 cycles. This is the key primitive for high-performance matmul and convolution kernels on Apple Silicon.

The memory hierarchy matters enormously for kernel optimization. **Threadgroup memory** (32 KB per threadgroup, ~5 cycle latency) should be used for tiled data loading in convolution and matmul. **Device memory** (~500 cycle latency) is the bottleneck — the arithmetic intensity threshold for compute-bound operations on Apple Silicon is approximately **19 FLOP/byte**. Apple also recommends preferring **SIMD shuffle operations** over threadgroup memory where possible, as Apple GPUs invest heavily in shuffle bandwidth.

Use **function constants** (`[[function_constant(N)]]`) for compile-time specialization of kernel parameters like filter size, stride, padding mode, and activation type. This eliminates runtime branching and enables the Metal compiler to aggressively optimize each variant. Cache the resulting `MTLComputePipelineState` objects in a dictionary at initialization.

---

## Practical architecture for the training pipeline

Putting it all together, here is the recommended architecture for the library's GPU backend:

**Memory layer**: All tensors backed by `MTLBuffer` with `.storageModeShared`. A `BufferPool` manages allocation with size-bucketed caching and LRU eviction. All layer buffers (weights, activations, gradients, optimizer state) are pre-allocated at network construction time.

**Compute layer**: Two paths — `MPSGraph` for standard operations (convolution, matmul, normalization, their gradients) with automatic differentiation and operation fusion, and custom Metal compute kernels for specialized operations. All compute pipeline states cached at initialization.

**Execution layer**: A single `MTLCommandBuffer` encodes the entire forward pass, backward pass, and weight update. No CPU round-trips between layers. Triple buffering for input batch data. A dispatch policy routes operations smaller than ~1K elements to Accelerate on CPU, everything else to GPU.

**Critical implementation details**: Wrap every training iteration in `autoreleasepool { }` — without this, temporary Metal objects leak and memory grows unboundedly. Use `commandBuffer.addCompletedHandler` for async GPU time measurement (`cb.gpuEndTime - cb.gpuStartTime`). Profile with Metal System Trace in Instruments to identify GPU idle gaps indicating synchronization problems. Never use `waitUntilCompleted()` between layers — only at batch boundaries when the CPU genuinely needs results.

For gradient computation specifically, encode backward kernels into the same command buffer as the forward pass. Each layer's backward method reads from stored activations and the incoming gradient tensor, writing to pre-allocated input gradient and weight gradient buffers. Gradient accumulation across micro-batches uses a simple additive kernel. The optimizer update (SGD, Adam) is encoded as the final compute dispatch in the command buffer.

---

## Conclusion

The path from "GPU shows no improvement" to significant acceleration requires three architectural changes, roughly in priority order. **First**, replace `ContiguousArray<Float>` with `MTLBuffer` as the Tensor's backing storage — this eliminates the per-operation copy overhead that almost certainly explains the previous poor results. **Second**, encode entire forward and backward passes into single command buffers, never synchronizing with the CPU between layers. **Third**, implement buffer pooling to avoid per-iteration allocation overhead.

Beyond these fundamentals, MPSGraph offers the highest-performance implementation of standard neural network operations with automatic differentiation and operation fusion — capabilities that would take months to replicate with custom kernels. Custom Metal shaders are worth writing only for novel operations or specific fused kernels not expressible in MPSGraph.

The unified memory architecture on Apple Silicon is a genuine advantage over discrete GPUs for ML training — but only if the software architecture exploits it. The zero-copy memory model means the CPU/GPU boundary should be nearly invisible in the Tensor abstraction, with the same buffer accessible via raw pointer (for Accelerate) or Metal encoder (for GPU compute). This is the design MLX uses, and it's the right model for a Swift neural network library targeting all Apple platforms.
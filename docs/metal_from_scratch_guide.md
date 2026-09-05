# Custom Metal Kernels from Scratch for Neural Network Training

**A complete guide to implementing GPU-accelerated neural network operations using pure Metal compute shaders — no MPS, no MPSGraph, no frameworks.**

---

## Why from scratch, and what it costs you

Building custom Metal kernels gives you full control over memory layout, kernel fusion, and dispatch strategy. The tradeoff is real: benchmarks show that custom matmul kernels typically achieve roughly 50% of MPS performance on M1 Max without months of per-chip tuning. But for a learning-focused library like Neuron, owning the entire stack means you understand every byte flowing through the GPU — and you can fuse operations in ways no general-purpose library anticipates.

The operations you need, in priority order: **matrix multiplication** (the backbone of everything), **im2col + GEMM convolution** (your primary bottleneck), **elementwise operations** (activations, bias add), **reductions** (normalization statistics), and **their backward passes**. Every other operation composes from these primitives.

---

## 1. MTLBuffer-backed Tensors: the foundation

Your `ContiguousArray<Float>` must become an `MTLBuffer`. On Apple Silicon's unified memory, a shared-mode buffer gives CPU and GPU zero-copy access to the same physical DRAM:

```swift
final class GPUTensor {
    let buffer: MTLBuffer
    let size: TensorSize  // rows, columns, depth
    
    var floatPointer: UnsafeMutablePointer<Float> {
        buffer.contents().assumingMemoryBound(to: Float.self)
    }
    
    var elementCount: Int {
        size.rows * size.columns * size.depth
    }
    
    init(device: MTLDevice, size: TensorSize) {
        let byteCount = size.rows * size.columns * size.depth * MemoryLayout<Float>.stride
        guard let buf = device.makeBuffer(length: byteCount, options: .storageModeShared) else {
            fatalError("Failed to allocate MTLBuffer of \(byteCount) bytes")
        }
        self.buffer = buf
        self.size = size
    }
    
    /// Initialize from existing data (zero-copy for CPU-side fills)
    init(device: MTLDevice, size: TensorSize, data: [Float]) {
        let byteCount = data.count * MemoryLayout<Float>.stride
        guard let buf = device.makeBuffer(bytes: data, length: byteCount, options: .storageModeShared) else {
            fatalError("Failed to allocate MTLBuffer")
        }
        self.buffer = buf
        self.size = size
    }
}
```

**Why `.storageModeShared`:** On Apple Silicon, this maps to unified memory — the CPU pointer and GPU access point to identical physical pages. No blit commands, no staging buffers, no copies. The `floatPointer` property lets you still use Accelerate/vDSP on the same memory for CPU fallback paths.

**Cache coherency rule:** GPU reads are only guaranteed to see CPU writes after the command buffer containing those reads is *committed after* the writes complete. Conversely, CPU reads after GPU writes require `waitUntilCompleted()` on the command buffer. Violating this causes silent data corruption.

**For small constants under 4 KB** (like convolution parameters, learning rate, dimensions): use `encoder.setBytes(&value, length:, index:)` instead of creating a buffer. Metal copies these inline into the command buffer.

---

## 2. Buffer pool: avoiding allocation in training loops

Every call to `device.makeBuffer()` involves a kernel-level allocation and potentially zero-filling. In a training loop, the same buffer sizes repeat every iteration. A buffer pool recycles freed buffers:

```swift
final class BufferPool {
    private let device: MTLDevice
    private var pools: [Int: [MTLBuffer]] = [:]  // size bucket -> available buffers
    
    init(device: MTLDevice) {
        self.device = device
    }
    
    /// Round up to nearest power of 2 for bucketing
    private func bucketSize(for byteCount: Int) -> Int {
        var size = 1
        while size < byteCount { size <<= 1 }
        return max(size, 256) // minimum 256 bytes
    }
    
    func acquire(byteCount: Int) -> MTLBuffer {
        let bucket = bucketSize(for: byteCount)
        if var available = pools[bucket], !available.isEmpty {
            let buf = available.removeLast()
            pools[bucket] = available
            return buf
        }
        return device.makeBuffer(length: bucket, options: .storageModeShared)!
    }
    
    func release(_ buffer: MTLBuffer) {
        let bucket = bucketSize(for: buffer.length)
        pools[bucket, default: []].append(buffer)
    }
    
    func drain() {
        pools.removeAll()
    }
}
```

MLX uses essentially this pattern — buffers are cached by size with LRU eviction when memory pressure rises. For training, pre-allocate all activation, gradient, and weight buffers at network construction time so the training loop itself does zero allocations.

---

## 3. Command buffer architecture: one buffer per training step

The single biggest performance mistake is creating separate command buffers per layer. Each `commit() + waitUntilCompleted()` round-trip costs 20–50 µs on macOS (~5x more on iOS). For a 10-layer network, that's 200–500 µs of pure overhead per pass.

The correct architecture encodes **all operations** — forward pass, backward pass, and weight update — into a single command buffer:

```swift
final class MetalEngine {
    let device: MTLDevice
    let queue: MTLCommandQueue
    let pool: BufferPool
    private var pipelines: [String: MTLComputePipelineState] = [:]
    
    init() {
        self.device = MTLCreateSystemDefaultDevice()!
        self.queue = device.makeCommandQueue()!
        self.pool = BufferPool(device: device)
    }
    
    /// Load and cache a compute pipeline
    func pipeline(named name: String) -> MTLComputePipelineState {
        if let cached = pipelines[name] { return cached }
        let library = device.makeDefaultLibrary()!
        let function = library.makeFunction(name: name)!
        let pipeline = try! device.makeComputePipelineState(function: function)
        pipelines[name] = pipeline
        return pipeline
    }
    
    /// Execute a full training step
    func trainStep(network: Network, inputBatch: GPUTensor, targets: GPUTensor) {
        guard let cmdBuffer = queue.makeCommandBuffer(),
              let encoder = cmdBuffer.makeComputeCommandEncoder() else { return }
        
        // Forward pass: all layers encoded sequentially
        var current = inputBatch
        for layer in network.layers {
            current = layer.encodeForward(encoder: encoder, input: current, engine: self)
        }
        
        // Loss computation
        let loss = encodeLoss(encoder: encoder, predictions: current, targets: targets)
        
        // Backward pass: reverse order
        var grad = loss.gradient
        for layer in network.layers.reversed() {
            grad = layer.encodeBackward(encoder: encoder, upstreamGrad: grad, engine: self)
        }
        
        // Weight updates
        for layer in network.layers {
            layer.encodeWeightUpdate(encoder: encoder, learningRate: 0.001, engine: self)
        }
        
        encoder.endEncoding()
        cmdBuffer.commit()
        // Only wait when you actually need results on CPU
        // cmdBuffer.waitUntilCompleted()
    }
}
```

**Why this works without barriers:** Within a single `MTLComputeCommandEncoder` using the default serial dispatch type, Metal guarantees that all memory writes from dispatch N are visible to dispatch N+1. So conv → batchnorm → relu chains work correctly without explicit `memoryBarrier()` calls.

**Triple buffering for input data:** Use 3 input batch buffers with a `DispatchSemaphore(value: 3)`. The CPU prepares batch N+1 while the GPU processes batch N. This keeps both processors busy.

---

## 4. Matrix multiplication kernel (the critical primitive)

Everything in neural networks reduces to matmul. Convolution via im2col becomes matmul. Dense layers are matmul. Even the backward passes are matmul variants. Getting this right is essential.

### 4a. Naive kernel (starting point)

```metal
#include <metal_stdlib>
using namespace metal;

// C[M×N] = A[M×K] × B[K×N]
kernel void matmul_naive(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    uint row = gid.y;  // M dimension
    uint col = gid.x;  // N dimension
    
    if (row >= M || col >= N) return;
    
    float sum = 0.0f;
    for (uint i = 0; i < K; i++) {
        sum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = sum;
}
```

This works but is catastrophically slow — every thread reads K elements from both A and B with no data reuse. Each element of A and B is loaded from device memory O(N) and O(M) times respectively.

### 4b. Tiled kernel with threadgroup memory

The key optimization: threads in a threadgroup cooperatively load tiles of A and B into fast threadgroup memory (~5 cycle latency vs ~500 for device memory), then compute from there:

```metal
#define TILE_SIZE 16

kernel void matmul_tiled(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid  [[thread_position_in_grid]],
    uint2 lid  [[thread_position_in_threadgroup]],
    uint2 tgid [[threadgroup_position_in_grid]]
) {
    // Threadgroup-shared tiles
    threadgroup float tileA[TILE_SIZE][TILE_SIZE];
    threadgroup float tileB[TILE_SIZE][TILE_SIZE];
    
    uint row = tgid.y * TILE_SIZE + lid.y;
    uint col = tgid.x * TILE_SIZE + lid.x;
    
    float sum = 0.0f;
    
    // Slide the tile window across the K dimension
    for (uint t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Cooperative load: each thread loads one element of each tile
        uint aCol = t * TILE_SIZE + lid.x;
        uint bRow = t * TILE_SIZE + lid.y;
        
        tileA[lid.y][lid.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[lid.y][lid.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        
        // Ensure all threads have loaded before computing
        threadgroup_barrier(mem_flags::mem_threadgroup);
        
        // Compute partial dot product from this tile
        for (uint i = 0; i < TILE_SIZE; i++) {
            sum += tileA[lid.y][i] * tileB[i][lid.x];
        }
        
        // Ensure all threads are done reading before next tile load
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

This reduces device memory reads by a factor of TILE_SIZE. With 16×16 tiles and 256 threads per threadgroup, this is a solid starting point.

### 4c. simdgroup_matrix kernel (Apple Silicon's "tensor core")

The `simdgroup_matrix<T, 8, 8>` type and `simdgroup_multiply_accumulate` intrinsic are Apple's hardware-accelerated matrix multiply. Each SIMD group (32 threads) processes 8×8 matrix tiles, performing 512 multiply-accumulates cooperatively. Available on Apple7+ (M1 and later):

```metal
#include <metal_stdlib>
using namespace metal;

// Uses simdgroup_matrix for hardware-accelerated 8x8 tiles
// Each simdgroup computes an 8×8 tile of the output
kernel void matmul_simdgroup(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C       [[buffer(2)]],
    constant uint& M      [[buffer(3)]],
    constant uint& N      [[buffer(4)]],
    constant uint& K      [[buffer(5)]],
    uint2 gid              [[thread_position_in_grid]],
    uint  simd_lane_id     [[thread_index_in_simdgroup]],
    uint  simd_group_id    [[simdgroup_index_in_threadgroup]],
    uint2 tgid             [[threadgroup_position_in_grid]]
) {
    // Each simdgroup handles an 8x8 output tile
    // Map simdgroup to output position
    const uint TILE = 8;
    
    // Number of simdgroups per threadgroup row
    // With 256 threads = 8 simdgroups, arrange in 2×4 grid
    const uint SG_PER_ROW = 4;
    uint sg_row = simd_group_id / SG_PER_ROW;
    uint sg_col = simd_group_id % SG_PER_ROW;
    
    uint out_row = tgid.y * (2 * TILE) + sg_row * TILE;  // 2 simdgroups vertically
    uint out_col = tgid.x * (SG_PER_ROW * TILE) + sg_col * TILE;  // 4 horizontally
    
    // Accumulator tile
    simdgroup_matrix<float, 8, 8> acc;
    acc = simdgroup_matrix<float, 8, 8>(0.0f);
    
    // Walk across K dimension in 8-wide steps
    for (uint k = 0; k < K; k += TILE) {
        simdgroup_matrix<float, 8, 8> mA;
        simdgroup_matrix<float, 8, 8> mB;
        
        // Load tile of A: rows [out_row..out_row+8], cols [k..k+8]
        simdgroup_load(mA, A + out_row * K + k, K);
        
        // Load tile of B: rows [k..k+8], cols [out_col..out_col+8]
        simdgroup_load(mB, B + k * N + out_col, N);
        
        // Hardware-accelerated multiply-accumulate
        simdgroup_multiply_accumulate(acc, mA, mB, acc);
    }
    
    // Store result
    if (out_row + TILE <= M && out_col + TILE <= N) {
        simdgroup_store(acc, C + out_row * N + out_col, N);
    }
}
```

**Dispatch setup for simdgroup matmul:**
```swift
let threadgroupSize = MTLSize(width: 32, height: 8, depth: 1) // 256 threads = 8 simdgroups
let gridSize = MTLSize(
    width: (N + 31) / 32,   // 4 simdgroups × 8 cols each = 32 cols per threadgroup
    height: (M + 15) / 16,  // 2 simdgroups × 8 rows each = 16 rows per threadgroup
    depth: 1
)
encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
```

**Key constraint:** `simdgroup_load` and `simdgroup_store` need the leading dimension (stride) to compute offsets for each lane. The pointer must point to the top-left corner of the 8×8 block, and the stride tells it how far apart rows are in memory. Unaligned edges of non-multiple-of-8 matrices need bounds checking — either pad your matrices to multiples of 8, or guard the loads.

---

## 5. Convolution via im2col + GEMM

For neural network convolution, the **implicit GEMM** approach (computing im2col indices on-the-fly) is best: it avoids the massive memory overhead of explicit im2col while mapping perfectly to your matmul kernel.

### 5a. The mathematical transformation

A 2D convolution with input `[N,C,H,W]`, filter `[K,C,kH,kW]`, output `[N,K,oH,oW]` can be expressed as:

```
im2col(input): [N*oH*oW, C*kH*kW]   (unfold patches into rows)
filter reshaped: [C*kH*kW, K]         (filters as columns)
output = im2col(input) × filter_reshaped → [N*oH*oW, K]
reshape to [N, K, oH, oW]
```

### 5b. Explicit im2col kernel

The simplest approach — unfold all input patches into a matrix, then call your matmul kernel:

```metal
// Unfolds input patches into columns for matrix multiplication
// Input: [N, C, H, W] (NCHW layout)
// Output: [N*oH*oW, C*kH*kW] (column matrix)
struct Conv2DParams {
    uint N;       // batch size
    uint C;       // input channels
    uint H;       // input height
    uint W;       // input width
    uint K;       // output channels (number of filters)
    uint kH;      // kernel height
    uint kW;      // kernel width
    uint oH;      // output height
    uint oW;      // output width
    uint strideH;
    uint strideW;
    uint padH;
    uint padW;
};

kernel void im2col(
    device const float* input  [[buffer(0)]],
    device float* columns      [[buffer(1)]],
    constant Conv2DParams& p   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.N * p.oH * p.oW;
    if (gid >= total) return;
    
    // Decode position
    uint n   = gid / (p.oH * p.oW);
    uint rem = gid % (p.oH * p.oW);
    uint oh  = rem / p.oW;
    uint ow  = rem % p.oW;
    
    uint col_width = p.C * p.kH * p.kW;
    uint col_row = gid;  // row in the column matrix
    
    // For each element in the patch
    for (uint c = 0; c < p.C; c++) {
        for (uint kh = 0; kh < p.kH; kh++) {
            for (uint kw = 0; kw < p.kW; kw++) {
                int ih = (int)(oh * p.strideH + kh) - (int)p.padH;
                int iw = (int)(ow * p.strideW + kw) - (int)p.padW;
                
                uint col_idx = c * p.kH * p.kW + kh * p.kW + kw;
                
                if (ih >= 0 && ih < (int)p.H && iw >= 0 && iw < (int)p.W) {
                    uint in_idx = n * p.C * p.H * p.W + c * p.H * p.W + ih * p.W + iw;
                    columns[col_row * col_width + col_idx] = input[in_idx];
                } else {
                    columns[col_row * col_width + col_idx] = 0.0f; // zero padding
                }
            }
        }
    }
}
```

Then the forward convolution is:
```
im2col(input) → columns[N*oH*oW, C*kH*kW]
matmul(columns, weights_reshaped[C*kH*kW, K]) → output[N*oH*oW, K]
reshape → output[N, K, oH, oW]
```

### 5c. Implicit GEMM (fused im2col + matmul)

The better approach: compute im2col indices *inside* the matmul kernel. No temporary matrix, no extra memory:

```metal
// Implicit GEMM: fuses im2col index computation into the matrix multiply
// Each thread computes one element of the output[N*oH*oW, K]
kernel void conv2d_implicit_gemm(
    device const float* input   [[buffer(0)]],   // [N, C, H, W]
    device const float* weights [[buffer(1)]],   // [K, C, kH, kW] stored as [K, C*kH*kW]
    device float* output        [[buffer(2)]],   // [N, K, oH, oW]
    constant Conv2DParams& p    [[buffer(3)]],
    device const float* bias    [[buffer(4)]],   // [K] or nullptr
    uint2 gid [[thread_position_in_grid]]
) {
    // gid.y = output spatial index (n*oH*oW), gid.x = output channel (k)
    uint spatial = gid.y;  // row in implicit matrix
    uint k = gid.x;       // which filter
    
    uint total_spatial = p.N * p.oH * p.oW;
    if (spatial >= total_spatial || k >= p.K) return;
    
    uint n   = spatial / (p.oH * p.oW);
    uint rem = spatial % (p.oH * p.oW);
    uint oh  = rem / p.oW;
    uint ow  = rem % p.oW;
    
    float sum = 0.0f;
    uint gemm_K = p.C * p.kH * p.kW;
    
    // Inner loop: accumulate over C*kH*kW (the implicit K dimension)
    for (uint ck = 0; ck < gemm_K; ck++) {
        // Decode im2col index on the fly
        uint c  = ck / (p.kH * p.kW);
        uint kr = (ck % (p.kH * p.kW)) / p.kW;
        uint kc = ck % p.kW;
        
        int ih = (int)(oh * p.strideH + kr) - (int)p.padH;
        int iw = (int)(ow * p.strideW + kc) - (int)p.padW;
        
        float a = 0.0f;
        if (ih >= 0 && ih < (int)p.H && iw >= 0 && iw < (int)p.W) {
            a = input[n * p.C * p.H * p.W + c * p.H * p.W + ih * p.W + iw];
        }
        
        // Weight layout: [K, C*kH*kW], accessing weights[k, ck]
        float w = weights[k * gemm_K + ck];
        sum += a * w;
    }
    
    // Add bias
    if (bias) sum += bias[k];
    
    // Store in NCHW output
    output[n * p.K * p.oH * p.oW + k * p.oH * p.oW + oh * p.oW + ow] = sum;
}
```

The tiled version of this is more complex — you'd load weight tiles into threadgroup memory and compute input indices on the fly — but the basic pattern is the same. This is exactly what cuDNN and MLX do internally.

### 5d. Dispatching convolution

```swift
func encodeConv2D(encoder: MTLComputeCommandEncoder,
                  input: GPUTensor, weights: GPUTensor, bias: GPUTensor?,
                  output: GPUTensor, params: Conv2DParams) {
    let pipeline = engine.pipeline(named: "conv2d_implicit_gemm")
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input.buffer, offset: 0, index: 0)
    encoder.setBuffer(weights.buffer, offset: 0, index: 1)
    encoder.setBuffer(output.buffer, offset: 0, index: 2)
    
    var p = params
    encoder.setBytes(&p, length: MemoryLayout<Conv2DParams>.stride, index: 3)
    
    if let bias = bias {
        encoder.setBuffer(bias.buffer, offset: 0, index: 4)
    }
    
    let totalSpatial = Int(params.N * params.oH * params.oW)
    let gridSize = MTLSize(width: Int(params.K), height: totalSpatial, depth: 1)
    let tgSize = MTLSize(width: min(Int(params.K), 16), 
                         height: min(totalSpatial, 16), depth: 1)
    
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
}
```

---

## 6. Elementwise operations (activations, bias add)

Elementwise operations are memory-bound — the GPU spends most of its time reading and writing, not computing. The key optimization is **fusing** them with adjacent operations (encode relu right after conv in the same command buffer). But you still need standalone kernels:

```metal
// ReLU forward: out[i] = max(0, x[i])
kernel void relu_forward(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = max(input[gid], 0.0f);
}

// ReLU backward: grad_input[i] = upstream[i] * (input[i] > 0 ? 1 : 0)
kernel void relu_backward(
    device const float* input    [[buffer(0)]],  // original input (from forward)
    device const float* upstream [[buffer(1)]],  // dL/dOutput
    device float* grad_input     [[buffer(2)]],  // dL/dInput
    uint gid [[thread_position_in_grid]]
) {
    grad_input[gid] = input[gid] > 0.0f ? upstream[gid] : 0.0f;
}

// Leaky ReLU forward
kernel void leaky_relu_forward(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    constant float& alpha      [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = input[gid];
    output[gid] = x > 0.0f ? x : alpha * x;
}

// Sigmoid forward: out = 1 / (1 + exp(-x))
kernel void sigmoid_forward(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    output[gid] = 1.0f / (1.0f + metal::exp(-input[gid]));
}

// Sigmoid backward: grad_input = upstream * out * (1 - out)
kernel void sigmoid_backward(
    device const float* sigmoid_output [[buffer(0)]],  // cached from forward
    device const float* upstream       [[buffer(1)]],
    device float* grad_input           [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    float s = sigmoid_output[gid];
    grad_input[gid] = upstream[gid] * s * (1.0f - s);
}

// GeLU forward (tanh approximation)
kernel void gelu_forward(
    device const float* input  [[buffer(0)]],
    device float* output       [[buffer(1)]],
    uint gid [[thread_position_in_grid]]
) {
    float x = input[gid];
    float cdf = 0.5f * (1.0f + metal::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
    output[gid] = x * cdf;
}

// Fused conv bias + activation (example of kernel fusion)
kernel void bias_relu_fused(
    device float* data          [[buffer(0)]],  // in-place on conv output
    device const float* bias    [[buffer(1)]],  // [K]
    constant uint& K            [[buffer(2)]],  // number of channels
    constant uint& spatial_size [[buffer(3)]],  // oH * oW
    uint gid [[thread_position_in_grid]]
) {
    // NCHW layout: channel = (gid / spatial_size) % K
    uint c = (gid / spatial_size) % K;
    float val = data[gid] + bias[c];
    data[gid] = max(val, 0.0f);  // fused bias + ReLU
}
```

**Dispatch for elementwise:**
```swift
let threadgroupSize = MTLSize(width: 256, height: 1, depth: 1)
let gridSize = MTLSize(width: tensor.elementCount, height: 1, depth: 1)
encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
```

---

## 7. Reduction kernels (for normalization)

Batch normalization and layer normalization require computing mean and variance across specific dimensions. This is a **parallel reduction** — the hardest primitive to write efficiently.

### 7a. SIMD-group reduction (building block)

```metal
// Reduce within a simdgroup (32 threads) using shuffles — 
// faster than threadgroup memory on Apple Silicon
inline float simd_sum(float val) {
    // Apple GPUs have SIMD width 32
    for (ushort offset = 16; offset > 0; offset >>= 1) {
        val += simd_shuffle_xor(val, offset);
    }
    return val;
}
```

### 7b. BatchNorm forward kernel

BatchNorm computes mean and variance per channel across the batch and spatial dimensions. With NCHW layout and input `[N, C, H, W]`, each channel c has `N * H * W` elements to reduce:

```metal
struct BNParams {
    uint N;   // batch size
    uint C;   // channels
    uint HW;  // H * W (spatial size)
    float epsilon;
    uint training;  // 1 for training, 0 for inference
    float momentum; // for running stats
};

// Phase 1: Compute per-channel mean and variance
// One threadgroup per channel
kernel void batchnorm_compute_stats(
    device const float* input    [[buffer(0)]],   // [N, C, H, W]
    device float* mean_out       [[buffer(1)]],   // [C]
    device float* var_out        [[buffer(2)]],   // [C]
    constant BNParams& p         [[buffer(3)]],
    uint tgid [[threadgroup_position_in_grid]],   // channel index
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint c = tgid;
    if (c >= p.C) return;
    
    uint count = p.N * p.HW;
    
    // Each thread accumulates a partial sum across its assigned elements
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (uint i = lid; i < count; i += tg_size) {
        uint n = i / p.HW;
        uint hw = i % p.HW;
        uint idx = n * p.C * p.HW + c * p.HW + hw;
        float val = input[idx];
        sum += val;
        sum_sq += val * val;
    }
    
    // Reduce within simdgroup
    sum = simd_sum(sum);
    sum_sq = simd_sum(sum_sq);
    
    // Reduce across simdgroups using threadgroup memory
    threadgroup float tg_sum[8];     // max 8 simdgroups in 256-thread tg
    threadgroup float tg_sum_sq[8];
    
    uint sg_id = lid / 32;  // simdgroup index
    uint sg_lane = lid % 32;
    
    if (sg_lane == 0) {
        tg_sum[sg_id] = sum;
        tg_sum_sq[sg_id] = sum_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lid == 0) {
        float total_sum = 0.0f;
        float total_sum_sq = 0.0f;
        uint num_sg = (tg_size + 31) / 32;
        for (uint i = 0; i < num_sg; i++) {
            total_sum += tg_sum[i];
            total_sum_sq += tg_sum_sq[i];
        }
        float mean = total_sum / float(count);
        float var = total_sum_sq / float(count) - mean * mean;
        mean_out[c] = mean;
        var_out[c] = var;
    }
}

// Phase 2: Normalize using computed stats
kernel void batchnorm_normalize(
    device const float* input    [[buffer(0)]],
    device float* output         [[buffer(1)]],
    device const float* mean     [[buffer(2)]],   // [C]
    device const float* var      [[buffer(3)]],   // [C]
    device const float* gamma    [[buffer(4)]],   // [C] scale
    device const float* beta     [[buffer(5)]],   // [C] shift
    constant BNParams& p         [[buffer(6)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.N * p.C * p.HW;
    if (gid >= total) return;
    
    // Decode channel from NCHW layout
    uint c = (gid / p.HW) % p.C;
    
    float x = input[gid];
    float m = mean[c];
    float v = var[c];
    float inv_std = 1.0f / metal::sqrt(v + p.epsilon);
    
    output[gid] = gamma[c] * (x - m) * inv_std + beta[c];
}
```

### 7c. BatchNorm backward kernel

The backward pass for batchnorm is notoriously complex. Three gradients are needed: dL/d(input), dL/d(gamma), dL/d(beta).

```metal
// Phase 1: Compute dL/dgamma and dL/dbeta (reductions over N,H,W per channel)
kernel void batchnorm_backward_params(
    device const float* input       [[buffer(0)]],   // [N, C, H, W]
    device const float* upstream    [[buffer(1)]],   // dL/dOutput [N, C, H, W]
    device const float* mean        [[buffer(2)]],   // [C]
    device const float* var         [[buffer(3)]],   // [C]
    device float* dgamma            [[buffer(4)]],   // [C]
    device float* dbeta             [[buffer(5)]],   // [C]
    constant BNParams& p            [[buffer(6)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint lid  [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]]
) {
    uint c = tgid;
    if (c >= p.C) return;
    
    uint count = p.N * p.HW;
    float m = mean[c];
    float inv_std = 1.0f / metal::sqrt(var[c] + p.epsilon);
    
    float sum_dg = 0.0f;  // dL/dgamma accumulator
    float sum_db = 0.0f;  // dL/dbeta accumulator
    
    for (uint i = lid; i < count; i += tg_size) {
        uint n = i / p.HW;
        uint hw = i % p.HW;
        uint idx = n * p.C * p.HW + c * p.HW + hw;
        
        float x_hat = (input[idx] - m) * inv_std;
        float dy = upstream[idx];
        
        sum_dg += dy * x_hat;
        sum_db += dy;
    }
    
    sum_dg = simd_sum(sum_dg);
    sum_db = simd_sum(sum_db);
    
    threadgroup float tg_dg[8];
    threadgroup float tg_db[8];
    uint sg_id = lid / 32;
    uint sg_lane = lid % 32;
    
    if (sg_lane == 0) {
        tg_dg[sg_id] = sum_dg;
        tg_db[sg_id] = sum_db;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    if (lid == 0) {
        float total_dg = 0.0f, total_db = 0.0f;
        uint num_sg = (tg_size + 31) / 32;
        for (uint i = 0; i < num_sg; i++) {
            total_dg += tg_dg[i];
            total_db += tg_db[i];
        }
        dgamma[c] = total_dg;
        dbeta[c] = total_db;
    }
}

// Phase 2: Compute dL/dInput
// dL/dx_i = (1/count) * gamma * inv_std * (count*dy_i - dbeta - x_hat_i*dgamma)
kernel void batchnorm_backward_input(
    device const float* input       [[buffer(0)]],
    device const float* upstream    [[buffer(1)]],
    device float* grad_input        [[buffer(2)]],
    device const float* mean        [[buffer(3)]],
    device const float* var         [[buffer(4)]],
    device const float* gamma       [[buffer(5)]],
    device const float* dgamma      [[buffer(6)]],
    device const float* dbeta       [[buffer(7)]],
    constant BNParams& p            [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.N * p.C * p.HW;
    if (gid >= total) return;
    
    uint c = (gid / p.HW) % p.C;
    float count = float(p.N * p.HW);
    float m = mean[c];
    float inv_std = 1.0f / metal::sqrt(var[c] + p.epsilon);
    
    float x_hat = (input[gid] - m) * inv_std;
    float dy = upstream[gid];
    
    grad_input[gid] = (gamma[c] * inv_std / count) * 
                       (count * dy - dbeta[c] - x_hat * dgamma[c]);
}
```

**LayerNorm** follows the same pattern but reduces over the feature dimension instead of the batch dimension. The kernel structure is identical — just change which indices you reduce over.

---

## 8. Backward passes for convolution and matmul

### 8a. Convolution backward

The conv2d backward pass has two components:

**Gradient w.r.t. input** (`dL/dInput`): This is a "full" convolution of the upstream gradient with the *rotated* (180°-flipped) weights. Mathematically:
```
dL/dInput = conv2d(dL/dOutput, rotate180(weights), padding=kH-1-padH, kW-1-padW)
```

Using im2col + GEMM:
```
im2col(dL/dOutput) → columns[N*H*W, K*kH*kW]
matmul(columns, weights_rotated_reshaped[K*kH*kW, C]) → grad_input[N*H*W, C]
```

Or equivalently — transpose the weight matrix and do the GEMM differently:
```
// dL/dInput = dL/dOutput_columns × weights^T
// where dL/dOutput is im2col'd and weights^T is [K, C*kH*kW]^T = [C*kH*kW, K]

// Actually simpler: col2im of (weights^T × dL/dOutput_reshaped)
```

A practical approach is to write a dedicated backward kernel:

```metal
// Convolution backward w.r.t. input
// For each input position (n,c,h,w), sum contributions from all output positions
// that used this input position in the forward pass
kernel void conv2d_backward_input(
    device const float* grad_output [[buffer(0)]],  // [N, K, oH, oW]
    device const float* weights     [[buffer(1)]],  // [K, C, kH, kW]
    device float* grad_input        [[buffer(2)]],  // [N, C, H, W]
    constant Conv2DParams& p        [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.N * p.C * p.H * p.W;
    if (gid >= total) return;
    
    uint n  = gid / (p.C * p.H * p.W);
    uint c  = (gid / (p.H * p.W)) % p.C;
    uint h  = (gid / p.W) % p.H;
    uint w  = gid % p.W;
    
    float sum = 0.0f;
    
    // For each filter and each kernel position that could have used (h,w)
    for (uint k = 0; k < p.K; k++) {
        for (uint kh = 0; kh < p.kH; kh++) {
            for (uint kw = 0; kw < p.kW; kw++) {
                // Which output position used input (h,w) at kernel offset (kh,kw)?
                int oh_check = (int)h + (int)p.padH - (int)kh;
                int ow_check = (int)w + (int)p.padW - (int)kw;
                
                // Must be divisible by stride
                if (oh_check % (int)p.strideH != 0) continue;
                if (ow_check % (int)p.strideW != 0) continue;
                
                int oh = oh_check / (int)p.strideH;
                int ow = ow_check / (int)p.strideW;
                
                if (oh >= 0 && oh < (int)p.oH && ow >= 0 && ow < (int)p.oW) {
                    float go = grad_output[n*p.K*p.oH*p.oW + k*p.oH*p.oW + oh*p.oW + ow];
                    float wt = weights[k*p.C*p.kH*p.kW + c*p.kH*p.kW + kh*p.kW + kw];
                    sum += go * wt;
                }
            }
        }
    }
    
    grad_input[gid] = sum;
}
```

**Gradient w.r.t. weights** (`dL/dWeights`): This is a cross-correlation between the input and the upstream gradient:
```
dL/dWeights[k,c,kh,kw] = sum over (n,oh,ow) of:
    input[n, c, oh*strideH+kh-padH, ow*strideW+kw-padW] * grad_output[n, k, oh, ow]
```

```metal
// Convolution backward w.r.t. weights
// One thread per weight element
kernel void conv2d_backward_weights(
    device const float* input       [[buffer(0)]],  // [N, C, H, W]
    device const float* grad_output [[buffer(1)]],  // [N, K, oH, oW]
    device float* grad_weights      [[buffer(2)]],  // [K, C, kH, kW]
    constant Conv2DParams& p        [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total_weights = p.K * p.C * p.kH * p.kW;
    if (gid >= total_weights) return;
    
    uint k  = gid / (p.C * p.kH * p.kW);
    uint c  = (gid / (p.kH * p.kW)) % p.C;
    uint kh = (gid / p.kW) % p.kH;
    uint kw = gid % p.kW;
    
    float sum = 0.0f;
    
    // Sum over batch and output spatial dimensions
    for (uint n = 0; n < p.N; n++) {
        for (uint oh = 0; oh < p.oH; oh++) {
            for (uint ow = 0; ow < p.oW; ow++) {
                int ih = (int)(oh * p.strideH + kh) - (int)p.padH;
                int iw = (int)(ow * p.strideW + kw) - (int)p.padW;
                
                if (ih >= 0 && ih < (int)p.H && iw >= 0 && iw < (int)p.W) {
                    float inp = input[n*p.C*p.H*p.W + c*p.H*p.W + ih*p.W + iw];
                    float go  = grad_output[n*p.K*p.oH*p.oW + k*p.oH*p.oW + oh*p.oW + ow];
                    sum += inp * go;
                }
            }
        }
    }
    
    grad_weights[gid] = sum;
}
```

### 8b. Using matmul for backward passes (the efficient way)

Rather than writing specialized backward kernels, you can express both backward passes as matrix multiplications — reusing your optimized matmul kernel:

**Weight gradient via GEMM:**
```
// columns = im2col(input) → [N*oH*oW, C*kH*kW]
// grad_output_reshaped → [N*oH*oW, K]
// dL/dWeights = columns^T × grad_output_reshaped → [C*kH*kW, K]
matmul_transA(columns, grad_output_reshaped) → grad_weights
```

**Input gradient via GEMM:**
```
// grad_output_reshaped → [N*oH*oW, K]
// weights_reshaped → [C*kH*kW, K]
// grad_columns = grad_output_reshaped × weights_reshaped^T → [N*oH*oW, C*kH*kW]
// col2im(grad_columns) → grad_input[N, C, H, W]
matmul_transB(grad_output_reshaped, weights_reshaped) → grad_columns
col2im(grad_columns) → grad_input
```

You'll need a `col2im` kernel (inverse of im2col) and transposed variants of your matmul:

```metal
// col2im: scatter-add column data back to image layout
kernel void col2im(
    device const float* columns  [[buffer(0)]],
    device float* output         [[buffer(1)]],  // [N, C, H, W] - must be zeroed first
    constant Conv2DParams& p     [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = p.N * p.oH * p.oW;
    if (gid >= total) return;
    
    uint n   = gid / (p.oH * p.oW);
    uint rem = gid % (p.oH * p.oW);
    uint oh  = rem / p.oW;
    uint ow  = rem % p.oW;
    
    uint col_width = p.C * p.kH * p.kW;
    
    for (uint c = 0; c < p.C; c++) {
        for (uint kh = 0; kh < p.kH; kh++) {
            for (uint kw = 0; kw < p.kW; kw++) {
                int ih = (int)(oh * p.strideH + kh) - (int)p.padH;
                int iw = (int)(ow * p.strideW + kw) - (int)p.padW;
                
                if (ih >= 0 && ih < (int)p.H && iw >= 0 && iw < (int)p.W) {
                    uint col_idx = c * p.kH * p.kW + kh * p.kW + kw;
                    uint out_idx = n * p.C * p.H * p.W + c * p.H * p.W + ih * p.W + iw;
                    // NOTE: Multiple threads may write to same out_idx
                    // Use atomic_fetch_add_explicit for correctness
                    // or restructure to avoid conflicts
                    atomic_fetch_add_explicit(
                        (device atomic_float*)&output[out_idx],
                        columns[gid * col_width + col_idx],
                        memory_order_relaxed
                    );
                }
            }
        }
    }
}
```

---

## 9. Transposed convolution (ConvTranspose2d)

Transposed convolution (deconvolution) is mathematically the gradient of convolution w.r.t. its input. So the `conv2d_backward_input` kernel above *is* a transposed convolution. To implement it as a standalone forward pass:

```metal
// TransposedConv2D forward = Conv2D backward w.r.t. input
// Input: [N, C_in, H_in, W_in]
// Weights: [C_in, C_out, kH, kW] (note: transposed from regular conv)
// Output: [N, C_out, H_out, W_out]
// H_out = (H_in - 1) * stride - 2*pad + kH + output_padding
kernel void conv_transpose2d_forward(
    device const float* input   [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant Conv2DParams& p    [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    // Same structure as conv2d_backward_input but with different parameter mapping
    // Each thread computes one output element
    uint total = p.N * p.K * p.oH * p.oW;  // K = C_out for transposed
    if (gid >= total) return;
    
    uint n   = gid / (p.K * p.oH * p.oW);
    uint k   = (gid / (p.oH * p.oW)) % p.K;
    uint oh  = (gid / p.oW) % p.oH;
    uint ow  = gid % p.oW;
    
    float sum = 0.0f;
    
    for (uint c = 0; c < p.C; c++) {
        for (uint kh = 0; kh < p.kH; kh++) {
            for (uint kw = 0; kw < p.kW; kw++) {
                int ih_check = (int)oh + (int)p.padH - (int)kh;
                int iw_check = (int)ow + (int)p.padW - (int)kw;
                
                if (ih_check % (int)p.strideH != 0) continue;
                if (iw_check % (int)p.strideW != 0) continue;
                
                int ih = ih_check / (int)p.strideH;
                int iw = iw_check / (int)p.strideW;
                
                if (ih >= 0 && ih < (int)p.H && iw >= 0 && iw < (int)p.W) {
                    float inp = input[n*p.C*p.H*p.W + c*p.H*p.W + ih*p.W + iw];
                    // Weight layout for transposed: [C_in, C_out, kH, kW]
                    float wt = weights[c*p.K*p.kH*p.kW + k*p.kH*p.kW + kh*p.kW + kw];
                    sum += wt * inp;
                }
            }
        }
    }
    
    output[gid] = sum;
}
```

---

## 10. Optimizer kernels (SGD, Adam)

Weight updates are elementwise and encode directly into the same command buffer:

```metal
// SGD with momentum
kernel void sgd_update(
    device float* param          [[buffer(0)]],
    device const float* grad     [[buffer(1)]],
    device float* velocity       [[buffer(2)]],
    constant float& lr           [[buffer(3)]],
    constant float& momentum     [[buffer(4)]],
    constant float& weight_decay [[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    float g = grad[gid] + weight_decay * param[gid];
    float v = momentum * velocity[gid] + g;
    velocity[gid] = v;
    param[gid] -= lr * v;
}

// Adam optimizer
kernel void adam_update(
    device float* param     [[buffer(0)]],
    device const float* grad[[buffer(1)]],
    device float* m         [[buffer(2)]],   // first moment
    device float* v         [[buffer(3)]],   // second moment
    constant float& lr      [[buffer(4)]],
    constant float& beta1   [[buffer(5)]],
    constant float& beta2   [[buffer(6)]],
    constant float& epsilon [[buffer(7)]],
    constant float& t       [[buffer(8)]],   // timestep (for bias correction)
    uint gid [[thread_position_in_grid]]
) {
    float g = grad[gid];
    
    // Update biased moments
    float m_new = beta1 * m[gid] + (1.0f - beta1) * g;
    float v_new = beta2 * v[gid] + (1.0f - beta2) * g * g;
    m[gid] = m_new;
    v[gid] = v_new;
    
    // Bias correction
    float m_hat = m_new / (1.0f - metal::pow(beta1, t));
    float v_hat = v_new / (1.0f - metal::pow(beta2, t));
    
    param[gid] -= lr * m_hat / (metal::sqrt(v_hat) + epsilon);
}
```

---

## 11. Function constants for kernel specialization

Use Metal function constants to create compile-time specialized kernel variants. This eliminates runtime branching and allows the compiler to optimize aggressively:

```metal
// Declare function constants
constant bool has_bias     [[function_constant(0)]];
constant bool use_relu     [[function_constant(1)]];
constant uint kernel_size  [[function_constant(2)]];

kernel void conv2d_specialized(
    device const float* input   [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output        [[buffer(2)]],
    constant Conv2DParams& p    [[buffer(3)]],
    device const float* bias    [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    // ... compute convolution ...
    float result = sum;
    
    if (has_bias) {  // compiled away when has_bias=false
        result += bias[k];
    }
    if (use_relu) {  // compiled away when use_relu=false
        result = max(result, 0.0f);
    }
    
    output[out_idx] = result;
}
```

On the Swift side, create specialized pipeline variants:
```swift
func makeSpecializedPipeline(hasBias: Bool, useRelu: Bool, kernelSize: UInt32) -> MTLComputePipelineState {
    let constants = MTLFunctionConstantValues()
    var bias = hasBias
    var relu = useRelu
    var ks = kernelSize
    constants.setConstantValue(&bias, type: .bool, index: 0)
    constants.setConstantValue(&relu, type: .bool, index: 1)
    constants.setConstantValue(&ks, type: .uint, index: 2)
    
    let function = try! library.makeFunction(name: "conv2d_specialized", constantValues: constants)
    return try! device.makeComputePipelineState(function: function)
}
```

Cache these at network initialization time in a dictionary keyed by the parameter combination.

---

## 12. Threadgroup size guidelines for Apple Silicon

All Apple Silicon GPUs use **SIMD width 32**. The hardware rules:

| Parameter | Guideline |
|-----------|-----------|
| Threadgroup size | Always a multiple of 32 |
| Sweet spot | 256 threads (8 SIMD groups) |
| Maximum | 1024 (but register-heavy kernels may be limited to 512) |
| Always check | `pipeline.maxTotalThreadsPerThreadgroup` at runtime |
| Threadgroup memory | 32 KB per threadgroup max |

**For matmul:** 256 threads arranged as 8 simdgroups computing 8×8 tiles = 16×32 output tile per threadgroup.

**For elementwise:** 256×1×1 threads, 1D grid covering all elements.

**For reductions:** 256 threads per threadgroup, one threadgroup per reduction group (per channel for batchnorm).

**For convolution:** The optimal arrangement depends on output dimensions — experiment with (16,16,1) and (8,32,1) threadgroup shapes.

Use `dispatchThreads(_:threadsPerThreadgroup:)` (non-uniform) rather than `dispatchThreadgroups` — it handles edge cases where the grid isn't a multiple of the threadgroup size.

---

## 13. Putting it all together: training loop

```swift
func train(network: Network, dataset: Dataset, epochs: Int) {
    let engine = MetalEngine()
    
    // Pre-allocate all buffers
    network.allocateBuffers(device: engine.device, pool: engine.pool)
    
    // Triple-buffer for input batches
    let semaphore = DispatchSemaphore(value: 3)
    var batchBuffers = (0..<3).map { _ in 
        engine.device.makeBuffer(length: batchByteSize, options: .storageModeShared)! 
    }
    var batchIndex = 0
    
    for epoch in 0..<epochs {
        for batch in dataset.batches {
            semaphore.wait()
            
            // CPU: fill next batch buffer
            let inputBuffer = batchBuffers[batchIndex % 3]
            memcpy(inputBuffer.contents(), batch.data, batch.byteCount)
            batchIndex += 1
            
            // GPU: process
            autoreleasepool {
                guard let cmdBuffer = engine.queue.makeCommandBuffer(),
                      let encoder = cmdBuffer.makeComputeCommandEncoder() else { return }
                
                // Encode entire forward + backward + update
                network.encodeTrainingStep(
                    encoder: encoder, 
                    input: inputBuffer, 
                    targets: batch.targets,
                    engine: engine
                )
                
                encoder.endEncoding()
                
                cmdBuffer.addCompletedHandler { _ in
                    semaphore.signal()
                }
                
                cmdBuffer.commit()
            }
        }
    }
}
```

**Critical details:**
- `autoreleasepool` around each iteration prevents Metal object leaks from growing memory unboundedly
- `addCompletedHandler` + semaphore enables async triple buffering
- Only call `waitUntilCompleted()` when you need to read results on CPU (e.g., logging loss)
- Use `cmdBuffer.gpuEndTime - cmdBuffer.gpuStartTime` for GPU-side timing

---

## 14. CPU/GPU routing policy

Not everything benefits from GPU dispatch. The overhead is ~20-50 µs per command buffer commit. Use this decision framework:

| Operation | GPU wins when | Use CPU (Accelerate) when |
|-----------|--------------|--------------------------|
| Matmul | M×K×N > ~128³ (2M elements) | Tiny matrices (< 64×64) |
| Convolution | Almost always for batch > 1 | 1×1 conv on tiny spatial dims |
| Elementwise | > ~10K elements | < 1K elements |
| Reduction | > ~1K elements per group | < 256 elements per group |

**But the key insight**: even if individual small ops would be faster on CPU, **the cost of breaking the GPU command buffer** (synchronizing, then resuming) often exceeds the per-operation savings. Keep everything on GPU within a single command buffer unless an operation is genuinely tiny (< ~1000 elements). The whole-pipeline throughput matters more than per-operation latency.

---

## 15. Debugging and profiling

**Metal GPU Capture:** In Xcode, use the Metal debugger to capture a frame and inspect every dispatch, buffer content, and execution time. This is your primary debugging tool.

**Metal System Trace** (Instruments): Shows GPU utilization over time. Look for gaps between kernel dispatches — these indicate synchronization overhead or CPU bottlenecks.

**Print debugging in Metal shaders:** Not directly possible. Write debug values into a dedicated output buffer, read back on CPU after `waitUntilCompleted()`.

**Common bugs:**
- **NaN values:** Check for division by zero in normalization (ensure epsilon is used), gradients exploding (add gradient clipping kernel)
- **Silent data corruption:** CPU wrote to buffer but command buffer was already committed — cache coherency violation
- **No GPU speedup:** Check if you're calling `waitUntilCompleted()` between layers instead of at batch boundaries
- **Memory growth:** Missing `autoreleasepool` or creating buffers inside training loop instead of pooling

---

## Summary: implementation priority order

1. **MTLBuffer-backed Tensor** — Replace ContiguousArray<Float> with shared-mode MTLBuffer
2. **Buffer pool** — Recycle allocations across training iterations  
3. **Single command buffer per step** — Encode all ops into one buffer
4. **Tiled matmul kernel** — The backbone of everything
5. **im2col + matmul convolution** — Forward pass using your matmul
6. **Elementwise kernels** — ReLU, sigmoid, GeLU (forward + backward)
7. **Reduction kernels** — BatchNorm/LayerNorm statistics
8. **Convolution backward kernels** — Weight gradient and input gradient via GEMM
9. **Optimizer kernels** — SGD, Adam
10. **Function constant specialization** — Fuse bias+activation into conv kernel
11. **simdgroup_matrix matmul** — Hardware-accelerated 8×8 tiles for peak performance

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Neuron is a Swift-based machine learning framework built from scratch for iOS, macOS, tvOS, and watchOS. It implements neural networks with custom backpropagation, supporting various architectures including CNNs, RNNs, LSTMs, GANs, and more. The framework runs on CPU with C-level optimizations via NumSwift.

## Build & Test Commands

### Building
```bash
swift build
```

### Testing
```bash
# Run all tests
swift test

# Run specific test
swift test --filter <TestName>
```

### Performance Note
Neuron runs ~10X faster in RELEASE mode due to compiler optimizations. For development:
```bash
swift build -c release
```

### Onboarding
Before development, install Xcode templates:
```bash
./scripts/onboard.sh
```

## Architecture

### Core Components

#### Tensor (Sources/Neuron/Tensor/)
- **Tensor**: The fundamental 3D array structure (`[[[Scalar]]]`) for all computations
- **TensorContext**: Holds backpropagation function for gradient computation
- **TensorSize**: Defines tensor dimensions as `(columns, rows, depth)`
- Supports automatic gradient calculation via `.gradients(delta:wrt:)` method
- Arithmetic operators overloaded for element-wise and tensor operations

#### Layers (Sources/Neuron/Layers/)
All layers inherit from `BaseLayer` and conform to the `Layer` protocol:
- **BaseLayer**: Base class handling batch processing, device management, weight initialization
- **EncodingType**: Enum defining all layer types for serialization
- Layer categories:
  - Convolutional: `Conv2d`, `TransConv2d`, `MaxPool`, `AvgPool`
  - Dense: `Dense`, `Flatten`, `Reshape`
  - Normalization: `BatchNormalize`, `LayerNormalize`
  - Activation: `ReLu`, `LeakyReLu`, `Sigmoid`, `Softmax`, `Tanh`, `Swish`, `SeLu`, `GeLu`
  - Regularization: `Dropout`
  - Recurrent: `LSTM`, `LSTMCell`
  - Other: `Embedding`

#### Trainable (Sources/Neuron/Trainable/)
- **Sequential**: Main network container that chains layers and manages forward/backward passes
- Implements result builder pattern: `Sequential { [Layer1(), Layer2(), ...] }`
- Handles automatic input size propagation through layers
- Supports model import/export via `.smodel` files

#### Optimizers (Sources/Neuron/Optimizers/)
- **BaseOptimizer**: Base class managing gradient application and metrics
- Available optimizers: `Adam`, `SGD`, `RMSProp`
- Features:
  - Learning rate decay via `DecayFunction` protocol (e.g., `ExponentialDecay`)
  - Gradient accumulation and normalization
  - Metrics reporting via `MetricsReporter`
  - L2 normalization support

#### Models (Sources/Neuron/Models/)
Pre-built training models:
- `Classifier`: Supervised learning with automatic batching and validation
- `GAN`, `WGAN`, `WGANGP`: Generative adversarial network variants
- `RNN`: Recurrent neural network wrapper

#### Devices (Sources/Neuron/Devices/)
- `CPU`: Default device (fully functional)
- `GPU`: Work in progress - Metal support is incomplete
- All layers and tensors can be assigned to devices

### Gradient System

Neuron uses a semi-automatic gradient system:
1. Each `Tensor` has a `context` with a `backpropagate` function
2. Tensors build computation graphs via `.setGraph(_:)`
3. Call `.gradients(delta:wrt:)` to compute gradients w.r.t. specific inputs
4. Returns `Tensor.Gradient` containing input, weight, and bias gradients
5. Supports multi-branch graphs with selective gradient computation

### First Layer Input Size
Only the first layer in a network requires explicit `inputSize` specification. All subsequent layers automatically calculate their input sizes when compiled by an `Optimizer`.

## Creating New Components

### New Layer
Follow the template in `.cursor/rules/layer.mdc`:
1. Inherit from `BaseLayer`
2. Add new case to `EncodingType` enum
3. Implement `forward(tensor:context:)` for the transformation
4. Implement Codable for serialization
5. Override `onInputSizeSet()` if weight initialization depends on input size

### New Optimizer
Follow the template in `.cursor/rules/optimizer.mdc`:
1. Inherit from `BaseOptimizer`
2. Implement `apply(_ gradients:)` with algorithm-specific logic
3. Maintain optimizer state (momentum, velocity, etc.) per layer
4. Call `build()` when trainable changes to reset state

### New Trainable
Follow the template in `.cursor/rules/trainable.mdc`:
1. Conform to `Trainable` and `Logger` protocols
2. Manage layer array and propagate `device`, `isTraining`, `batchSize`
3. Implement `compile()` to validate and connect layers
4. Implement `predict(_:context:)` for forward pass

## Code Style (from .cursor/rules/general.mdc)

- Use descriptive variable/function names
- Add comprehensive documentation comments
- Follow Swift naming conventions
- Use `public` for API, `private` for implementation
- Validate input dimensions and provide meaningful errors
- Use NumSwift operations for mathematical computations
- Minimize memory allocations in forward passes
- Create unit tests verifying gradient computations and serialization
- **Use `Tensor.Scalar` typealias**: All new functions that need access to a scalar type like `Float` should use `Tensor.Scalar` instead of hardcoding `Float`. This ensures compatibility with Float16 quantization (when `QUANTIZED_F16` flag is set). See `TensorSIMD.swift` for examples of this pattern.
- **Support both Float and Float16**: Any new math functions on a Tensor should be implemented for both `Float` and `Float16` types. This ensures the framework works correctly regardless of whether quantization is enabled.

## Dependencies

- **NumSwift**: C-optimized numeric operations (SIMD, BLAS-like functions)
- **Logger**: Logging framework for debugging
- **swift-numerics**: Apple's numerics library

## Branch Strategy

- `main`: Stable production branch
- `develop`: Development branch for integration
- Feature branches: Branch off `develop`, PR into `develop`
- Automated tests must pass before PR merge

## Important Notes

- No GPU execution yet - all operations run on CPU with multi-threading
- Use `RELEASE` scheme for performance benchmarks
- Tensor operations use Float (or Float16 with QUANTIZED_F16 flag)
- Model export/import uses `.smodel` format via `ExportHelper`
- MetricsReporter tracks loss, accuracy, and validation metrics during training
- NetworkContext carries batch processing metadata through forward passes

## Debugging Guide

### Common Issues and Debugging Patterns

#### Gradient Flow Issues
1. **Check TensorContext**: Each layer's `forward()` method should create a `TensorContext` with a proper `backpropagate` closure
2. **Verify Graph Building**: Use `.setGraph(_:)` to connect tensors in the computation graph
3. **Test Gradients**: Call `.gradients(delta:wrt:)` on output tensors to verify gradient computation
   - Pass `wrt:` parameter to get gradients w.r.t. specific inputs
   - Returns `Tensor.Gradient` with `.input`, `.weights`, `.biases` arrays
4. **Multi-branch Graphs**: When tensors have multiple inputs, set graph for each: `output.setGraph(input1); output.setGraph(input2)`

#### Shape Mismatches
- Tensors are always 3D: `[[[Scalar]]]` with shape `[columns, rows, depth]`
- First layer needs explicit `inputSize`; others auto-calculate from previous layer
- Use `tensor.shape` to inspect dimensions (returns `[Int]`)
- `Tensor.features` property is a hack for handling different array structures (see comment in Tensor.swift:82-87)

#### Training Issues
- **GradientAccumulator**: Collects and averages gradients across batch
  - Call `insert(_:)` to add gradients
  - Call `accumulate(clearAtEnd:)` to get averaged result
  - Set `.average = false` to disable averaging
- **Optimizer.fit()**: Returns `Output` tuple with `(outputs, gradients, loss, accuracy)`
- **Step Sequence**: `zeroGradients()` → forward pass → calculate loss → `apply(gradients)` → `step()`
- **Learning Rate Decay**: Set `optimizer.decayFunction` (e.g., `ExponentialDecay`) - automatically managed

#### Loss Function Selection
Match loss function to output layer:
- `crossEntropySoftmax` / `binaryCrossEntropySoftmax`: Use WITH Softmax layer (optimized derivative)
- `crossEntropy` / `binaryCrossEntropy`: Use WITHOUT Softmax layer
- `meanSquareError`: Regression tasks
- `wasserstein`: For WGAN variants

#### Metrics & Accuracy
- `MetricsReporter`:
  - Set `frequency` to control reporting interval (steps)
  - `receive` closure called with metrics dictionary
  - Tracks running totals internally (see `totalCorrectGuesses`, etc.)
  - Binary vs multi-class: Auto-detected based on comparison threshold (0.5 for binary)
- Accuracy calculation: Compares `indexOfMax` of predictions vs labels

#### Weight Initialization
- **InitializerType options**: `.heNormal` (default), `.heUniform`, `.xavierNormal`, `.xavierUniform`, `.normal(std:)`
- Formula: `heNormal = gaussian * sqrt(2/inputSize)`, `xavierNormal = gaussian * sqrt(2/(input+output))`
- Only first layer or layers with `inputs` parameter initialize weights immediately
- Others initialize when `inputSize` is set (via `onInputSizeSet()`)

#### Multithreading & Performance
- `Constants.maxWorkers`: Auto-detects performance cores (power-of-2 for even batch splits)
- `Device.qosPriority`: QoS priority for threading (CPU vs GPU)
- Batch processing: Uses `concurrentForEach(workers:priority:)` for parallel execution
- `NetworkContext`: Carries `batchRange`, `indexInBatch`, `threadId` through forward pass

#### Testing Patterns
From NeuronTests.swift:
1. Create layer with `inputSize`
2. Set weights manually for deterministic testing
3. Forward pass: `layer.forward(tensor: input)`
4. Build graph: `output.setGraph(input)`
5. Backward pass: `output.gradients(delta: errorTensor, wrt: input)`
6. Assert expected shapes and values

### Key Files for Debugging

- **Tensor.swift**: Core data structure, arithmetic, gradient computation
- **TensorContext.swift**: Backpropagation function wrapper (very simple!)
- **Gradient.swift**: GradientAccumulator for averaging batch gradients
- **Optimizer.swift**: Training loop, `fit()` method, gradient application
- **LossFunction.swift**: Loss calculations and derivatives
- **Metrics.swift**: Accuracy calculation logic, timer utilities
- **Layer.swift**: BaseLayer batch processing, device management

### Debugging Tensor Operations
```swift
// Inspect tensor
print(tensor.shape)        // [columns, rows, depth]
print(tensor.isEmpty)      // Check if empty
print(tensor.value)        // Raw 3D array

// Check computation graph
print(tensor.graph.keys)   // UUIDs of input tensors
print(tensor.graphChain)   // Set of all UUIDs in chain

// Manual gradient test
let output = layer.forward(tensor: input)
output.setGraph(input)
let error = Tensor(/* expected error */)
let grads = output.gradients(delta: error, wrt: input)
print(grads.input.count, grads.weights.count, grads.biases.count)
```

### Common Pitfalls
1. Forgetting to call `output.setGraph(input)` after forward pass
2. Using wrong loss function for activation layer (e.g., crossEntropy instead of crossEntropySoftmax)
3. Not calling `zeroGradients()` before training step
4. Shape mismatches: Remember all tensors are 3D internally
5. Not setting `isTraining = true` before training (affects Dropout, BatchNorm)
6. Accessing `learningRate` when `decayFunction` is set (use property, not field)
7. Not calling `optimizer.step()` after `apply()` (needed for decay function updates)

## Performance & Memory Profiling

### Timing Operations

The framework uses `Date().timeIntervalSince1970` for high-level timing. Pattern from Classifier.swift:64:

```swift
let startTime = Date().timeIntervalSince1970
// ... training code ...
print("----epoch \(i) completed: \(Date().timeIntervalSince1970 - startTime)s-----")
```

For more granular timing, use the built-in `MetricsReporter` timer system:

```swift
// In your optimizer or training loop
optimizer.metricsReporter?.startTimer(metric: .batchTime)
// ... batch processing ...
optimizer.metricsReporter?.endTimer(metric: .batchTime)

// Available timer metrics:
// - .batchTime: Time to process one batch (forward + backward + gradient calculation)
// - .optimizerRunTime: Time from zeroGradients() to step() (gradient application)
// - .batchConcurrency: Track concurrent batch processing

// Access results via receive closure
optimizer.metricsReporter?.receive = { metrics in
  print("Batch time: \(metrics[.batchTime] ?? 0)s")
  print("Optimizer time: \(metrics[.optimizerRunTime] ?? 0)s")
}
```

**Timer Implementation Details:**
- Timers stored in `[Metric: [Date]]` dictionary (Metrics.swift:139)
- Uses `timeIntervalSince1970` for calculations (Metrics.swift:181)
- Automatically averages multiple timer instances
- Thread-safe via `SynchronousOperationQueue` with barrier blocks

### Memory Management Best Practices

#### Array Capacity Management
The codebase uses `keepingCapacity: true` extensively to avoid reallocation:

```swift
// Gradient accumulator (Gradient.swift:24-26)
biasGradients.removeAll(keepingCapacity: true)
weightGradients.removeAll(keepingCapacity: true)
inputGradients.removeAll(keepingCapacity: true)

// Optimizer state reset (Adam.swift:179-182, RMSProp.swift:63-64, SGD.swift:92-93)
m.removeAll(keepingCapacity: true)
v.removeAll(keepingCapacity: true)

// Pre-allocate when size is known (MaxPool.swift:97)
currentIndicies.reserveCapacity(inputSize.depth)
```

**Key Pattern:** When clearing arrays that will be refilled, use `keepingCapacity: true` to avoid deallocating/reallocating memory.

#### Concurrent Processing
Batch operations use multi-threaded workers to maximize CPU usage:

```swift
// Pattern from WGANGP.swift:38
Array(0..<batchSize).concurrentForEach(workers: Constants.maxWorkers) { _, i in
  // Process batch item i
}

// Constants.maxWorkers (Constants.swift:11-19):
// - Auto-detects performance cores via sysctl "hw.perflevel0.physicalcpu"
// - Rounds down to nearest power of 2 for even batch splits
// - Default: 4 if detection fails
```

### Performance Profiling with Xcode Instruments

**Time Profiler Setup:**
1. Build with Release configuration: `swift build -c release`
2. Run with Instruments Time Profiler
3. Focus on hot paths:
   - `forward(tensor:context:)` methods in layers
   - NumSwift operations (matmul, convolution, etc.)
   - Gradient accumulation in `accumulate()`
   - `concurrentForEach` worker threads

**Allocations Instrument:**
- Watch for Tensor allocations in tight loops
- Check for unnecessary copies (use `.detached()` when gradient tracking not needed)
- Monitor GradientAccumulator array growth
- Look for Metal buffer allocations (if GPU path enabled)

**Key Optimization Points:**
1. **Dense layer matmul** (Dense.swift): Uses NumSwiftC transpose for performance
2. **Gradient accumulation**: Average only if `iterations > 1 && average == true` (Gradient.swift:78)
3. **Batch processing**: Split across `Constants.maxWorkers` threads
4. **Memory reuse**: Arrays cleared with `keepingCapacity: true`

### Memory Layout Inspection

For debugging tensor memory:
```swift
// Tensor size in bytes
let tensorByteSize = tensor.shape.reduce(1, *) * MemoryLayout<Tensor.Scalar>.stride

// Check scalar size (Float vs Float16)
print("Scalar size: \(MemoryLayout<Tensor.Scalar>.size) bytes")
// 4 bytes for Float, 2 bytes for Float16 (with QUANTIZED_F16)

// Metal buffer sizing (GPUManager.swift:171-174)
let dataBuffer = device.makeBuffer(bytes: &data,
                                   length: MemoryLayout<Tensor.Scalar>.stride * data.count,
                                   options: [])
```

### Performance Checklist

Before profiling, verify:
- [ ] Running in **Release mode** (`-c release` flag)
- [ ] `Constants.maxWorkers` matches your CPU cores
- [ ] Batch size is power of 2 (for optimal thread distribution)
- [ ] Not printing tensors in hot loops (huge performance hit)
- [ ] MetricsReporter frequency set appropriately (not every iteration)
- [ ] Arrays cleared with `keepingCapacity: true` where reused

### Common Performance Issues

1. **Slow training**: Not running in Release mode (10x slower in Debug)
2. **Memory growth**: Forgetting to call `zeroGradients()` / `gradientAccumulator.clear()`
3. **Thread contention**: Batch size not evenly divisible by `maxWorkers`
4. **Unnecessary gradient tracking**: Use `detatch: true` when generating fake samples (WGANGP.swift:43)
5. **MetricsReporter overhead**: Set high `frequency` value or disable unused metrics

### Profiling Example

```swift
// Time a specific operation
let start = Date().timeIntervalSince1970
let output = network.predict(input, context: .init())
let duration = Date().timeIntervalSince1970 - start
print("Forward pass: \(duration)s")

// Profile with MetricsReporter
let reporter = MetricsReporter(
  frequency: 10,
  metricsToGather: [.batchTime, .optimizerRunTime]
)
optimizer.metricsReporter = reporter
reporter.receive = { metrics in
  if let batchTime = metrics[.batchTime],
     let optimTime = metrics[.optimizerRunTime] {
    let inferenceTime = batchTime - optimTime
    print("Batch: \(batchTime)s | Optimizer: \(optimTime)s | Inference: \(inferenceTime)s")
  }
}

// Clear between epochs to reset running averages
reporter.totalCorrectGuesses = 0
reporter.totalGuesses = 0
```

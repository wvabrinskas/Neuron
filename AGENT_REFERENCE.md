# Neuron Framework – Agent Reference

This document provides a comprehensive reference for AI agents working with the Neuron codebase. Use it to understand the architecture, conventions, and critical implementation details.

---

## Project Overview

**Neuron** is a Swift-based machine learning framework for iOS, macOS, tvOS, and watchOS. It implements neural networks with custom backpropagation, supporting CNNs, RNNs, LSTMs, GANs, and more. The framework runs on CPU with C-level optimizations via NumSwift.

- **Repository**: `wvabrinskas/Neuron` on GitHub
- **Platforms**: iOS 14+, macOS 14+
- **Swift**: 5.10.0
- **Dependencies**: NumSwift, Logger, swift-numerics, swift-atomics

---

## Build & Test Commands

```bash
# Build
swift build

# Build (release – ~10x faster)
swift build -c release

# Run all tests (always use CI=true)
CI=true swift test

# Run specific test
CI=true swift test --filter <TestName>

# Verbose tests
CI=true swift test -v

# Onboarding (install Xcode templates)
./scripts/onboard.sh
```

---

## Architecture

### Tensor System

- **Tensor**: Fundamental type backed by flat `ContiguousArray<Scalar>` storage with `TensorSize` metadata
- **Layout**: `(columns, rows, depth)` – use `tensor.shape` for `[Int]`
- **TensorContext**: Holds backpropagation function for gradient computation
- **Gradient flow**: `.gradients(delta:wrt:)` returns `Tensor.Gradient` with `.input`, `.weights`, `.biases`
- **Graph**: Use `.setGraph(_:)` to connect tensors for backprop

### Layers (`Sources/Neuron/Layers/`)

All layers inherit from `BaseLayer`:

| Category      | Layers                                                                 |
|---------------|-----------------------------------------------------------------------|
| Convolutional | Conv2d, TransConv2d, MaxPool, AvgPool                                |
| Dense         | Dense, Flatten, Reshape                                               |
| Normalization | BatchNormalize, LayerNormalize, **InstanceNormalize**                 |
| Activation    | ReLu, LeakyReLu, Sigmoid, Softmax, Tanh, Swish, SeLu, GeLu            |
| Regularization| Dropout                                                               |
| Recurrent     | LSTM, LSTMCell                                                        |
| Other         | Embedding                                                             |

### Trainable & Optimizers

- **Sequential**: Main container, result builder pattern
- **Optimizers**: Adam, SGD, RMSProp
- **First layer**: Only the first layer needs explicit `inputSize`; others auto-calculate when compiled

### Devices

- **CPU**: Default, fully functional
- **GPU**: Work in progress (Metal support incomplete)

---

## Critical: Optimizer Gradient Layout

**Layers with multiple params (e.g. gamma, beta) must use the same layout for `weights` and gradients.**

The optimizer (e.g. Adam) applies weight decay with `weights[i]` for each gradient index `i`. If layouts differ, training is corrupted.

### Correct Pattern (InstanceNormalize)

```swift
// weights: beta | gamma (depth 0 = beta, depth 1 = gamma)
public override var weights: Tensor {
  get { return beta.concat(gamma, axis: 2) }
  set {}
}

// backward: return dBeta | dGamma (same order as weights)
return (dInput, dBetaTensor.concat(dGammaTensor, axis: 2), Tensor())

// apply: depthSlice(0) = beta, depthSlice(1) = gamma
let betaWeights = gradients.weights.depthSliceTensor(0)
let gammaWeights = gradients.weights.depthSliceTensor(1)
gamma = gamma - gammaWeights
beta = beta - betaWeights
```

### Wrong Pattern (causes training failure)

- `weights`: beta | gamma
- `gradients`: gamma | beta  ← **Mismatch!** Adam applies wrong values during weight decay

---

## InstanceNormalize Fix (Summary)

InstanceNormalize was breaking MNIST training due to a **gradient/weights layout mismatch**:

1. **Problem**: `weights` used `beta.concat(gamma)` but gradients returned `dGamma.concat(dBeta)` – opposite order
2. **Fix**: Return `dBeta.concat(dGamma)` from backward to match weights
3. **Fix**: In `apply()`, use `depthSlice(0)` for beta and `depthSlice(1)` for gamma
4. **Fix**: Simplify `weights` to return `beta.concat(gamma)` directly (no incorrect size metadata)

---

## Code Style & Conventions

- Use `Tensor.Scalar` instead of hardcoding `Float` (supports Float16 when `QUANTIZED_F16` is set)
- Implement both Float and Float16 for new math functions
- Use `public` for API, `private` for implementation
- Minimize allocations in forward passes
- Use NumSwift operations for math
- Add unit tests for gradient computation and serialization

---

## Key Files

| File | Purpose |
|------|---------|
| `Sources/Neuron/Tensor/Tensor.swift` | Core tensor, arithmetic, gradients |
| `Sources/Neuron/Tensor/TensorContext.swift` | Backpropagation wrapper |
| `Sources/Neuron/Gradient.swift` | GradientAccumulator |
| `Sources/Neuron/Optimizers/Optimizer.swift` | Base optimizer, fit loop |
| `Sources/Neuron/Optimizers/Adam.swift` | Adam optimizer, weight decay |
| `Sources/Neuron/Layers/Layer.swift` | BaseLayer, batch processing |
| `Sources/Neuron/Layers/InstanceNormalize.swift` | Instance normalization |
| `Sources/Neuron/Devices/CPU.swift` | CPU device |
| `Sources/Neuron/Devices/Devices.swift` | Device protocol |

---

## Gradient Flow

1. `zeroGradients()` → forward pass → loss → backward
2. `gradientAccumulator.insert(gradient)` – per batch item
3. `gradientAccumulator.accumulate()` – averages gradients
4. Optimizer `run()` applies Adam/SGD to each layer’s gradient
5. `layer.apply(gradients:learningRate:)` – subtracts updates from params

---

## Loss Functions

| Output layer | Loss function |
|--------------|---------------|
| With Softmax | `crossEntropySoftmax` / `binaryCrossEntropySoftmax` |
| Without Softmax | `crossEntropy` / `binaryCrossEntropy` |
| Regression | `meanSquareError` |
| WGAN | `wasserstein` |

---

## Common Pitfalls

1. Forgetting `output.setGraph(input)` after forward pass
2. Wrong loss function for activation (e.g. crossEntropy with Softmax)
3. Not calling `zeroGradients()` before training step
4. Shape mismatches: tensors are 3D internally
5. Not setting `isTraining = true` (affects Dropout, BatchNorm)
6. Not calling `optimizer.step()` after `apply()`
7. **Gradient/weights layout mismatch** for layers with multiple params

---

## Testing Pattern

```swift
let layer = SomeLayer(inputSize: input.shape.tensorSize)
let output = layer.forward(tensor: input)
output.setGraph(input)
let grads = output.gradients(delta: errorTensor, wrt: input)
XCTAssert(grads.input.first?.isEmpty == false)
XCTAssert(grads.weights.first!.isValueEqual(to: expected, accuracy: 0.00001))
```

---

## Package Structure

```
Neuron/
├── Package.swift
├── Sources/Neuron/
│   ├── Tensor/           # Tensor, TensorStorage, TensorMath
│   ├── Layers/            # All layer implementations
│   ├── Optimizers/        # Adam, SGD, RMSProp
│   ├── Trainable/         # Sequential
│   ├── Models/            # Classifier, GAN, WGAN, WGANGP, RNN
│   ├── Devices/           # CPU, GPU
│   ├── Gradient.swift     # GradientAccumulator
│   └── ...
├── Tests/NeuronTests/
└── ...
```

---

## Float16 Quantization

To enable Float16 mode, add to the Neuron target in `Package.swift`:

```swift
swiftSettings: [
  .define("QUANTIZED_F16")
]
```

Parent projects cannot pass compiler flags to SPM dependencies, so use a local clone.

---

## Debugging

```swift
// Inspect tensor
print(tensor.shape)    // [columns, rows, depth]
print(tensor.isEmpty)

// Check graph
print(tensor.graph.keys)   // UUIDs of inputs
print(tensor.graphChain)  // All UUIDs in chain

// Manual gradient test
let output = layer.forward(tensor: input)
output.setGraph(input)
let grads = output.gradients(delta: error, wrt: input)
print(grads.input.count, grads.weights.count, grads.biases.count)
```

---

*This reference was generated from the Neuron codebase and CLAUDE.md. Last updated: March 2025.*

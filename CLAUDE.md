# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Maintaining this document:** After making large changes to the codebase, update this file with references to those changes. Keep the architecture, debugging patterns, and common pitfalls sections current so future agents have accurate context.

**Document placement:** When creating agent references, migration summaries, or other markdown documentation, place them in the `docs/` directory rather than the project root.

**Recording trade-offs:** When you make a change that is correct but not ideal — a constraint forced a compromise, you took a shortcut you can defend but don't love, or you put something somewhere it doesn't conceptually belong — add an entry to `docs/LEARNINGS.md` rather than a `TODO` comment nobody will find. Each entry states what forced the decision, what the code does now, what it cost, and the trigger that should make someone revisit it. `LEARNINGS.md` is not a bug list (fix those) and not a changelog (that's Recent Changes below).

## Recent Changes (Reference for Updates)

- **BPE tokenizer + RNN next-token pipeline**: The RNN moved from `Vectorizer` (one index per character) to `BPETokenizer` (subword IDs), which broke several invariants at once. Fixed across `BPETokenizer`, `TokenizableDataset`, `RNN`, `LossFunction`, and `Metrics`:
  - **Token tensors are depth-major**: `tokenize` returns `rows: 1, columns: 1, depth: tokenCount`. `Embedding.forward` reads one index per *depth slice*, so packing IDs along `columns` leaves the tensor at `depth: 1` and the model only ever sees the first token — and `RNN.compile` then reads `shape[2] == 1` and builds the LSTM with `batchLength: 1`.
  - **`BPETokenizer` IDs are contiguous `0..<vocabSize`**: `nextId` is derived (`vocab.count`), not stored. Seeding it from `Vectorizer.lastKey + 1` skipped an ID, because `lastKey` is already the next free slot. `vocabSize` is `vocab.count` (merge tokens included); reporting only the base vocabulary let merge IDs index past the embedding table.
  - **Control tokens**: `padTokenId` / `bosTokenId` / `eosTokenId` / `controlTokenIds` on `Tokenizing`. `</w>` is deliberately *not* a control token — it decodes to a space. `decode(_:skipControlTokens:)` drops the rest so a padded sequence doesn't render as `<pad><pad><pad>`.
  - **Next-token pairs**: `TokenizableDataset.nextTokenPair(for:sequenceLength:)` wraps in `<bos>`/`<eos>`, shifts by one **token**, truncates at `sequenceLength + 1`, and pads. Shifting the *text* (`dropFirst()`) and re-encoding does not work — tokenization is not shift-equivariant, so every position after the first stays identical to the input and the model learns to copy.
  - **Sparse loss/accuracy are mask-aware**: `LossFunction.calculate(_:correct:ignoring:)` and `.derivative(_:correct:ignoring:)` exclude an index (the pad token) from loss and emit a zero gradient row for it; `calculateAccuracy` skips those timesteps. Wired through `Optimizer.ignoreLabelIndex`, which `RNN.readyUp` sets to `dataset.padTokenId`. Without it a padded batch is scored on predicting padding.
  - **Sparse labels need `sparse:` in accuracy**: a sparse label slice holds the class as its *value*, so `indexOfMax` is always `0`. `LossFunction.isSparse` drives this.
  - **`RNN.predict`**: selects the next token by argmax / `randomChoice` over the output distribution (it previously read `lastSlice.first`, a *probability*, as a token ID), appends the predicted ID rather than re-encoding decoded text, stops on `eosTokenId`, and decodes the accumulated IDs in **one pass** — per-token decoding trims `</w>` and loses every space.
  - **Generation slides a context window**: `LSTM.forward` iterates a fixed `0..<batchLength` from the *start* of its input and never reads past it (see the TODO at `LSTM.swift`). `RNN.predict` used to concatenate every predicted token onto the input, so once the sequence exceeded `wordLength` the context froze on the first window — the distribution stopped changing and generation repeated one token until `maxTokenCount` (default 20, commonly larger than `wordLength`). `predict` now rebuilds its input from `tokenIds.suffix(wordLength)` each step and reads the output slice for the last *real* timestep in that window.
  - **`RNN` import restores exact state**: `importFrom` no longer calls `readyUp()` before assigning the imported network. Doing so rebuilt the dataset (re-training the tokenizer and discarding the imported vocabulary) and compiled a throwaway randomly-initialised network that the imported one then replaced — leaving `wordLength`, `lstm`, and `embedding` describing layers no longer in the graph. `restore(network:)` now takes `vocabSize`/`padTokenId` from the imported tokenizer and recovers the sequence length from the imported layers (`Embedding.batchLength` / `LSTM.batchLength`, both now public), which is the only source available at import time. `readyUp()` skips `compile` when a network was imported and instead validates new data against it, failing loudly if the vocabulary drifted.
  - **`RNN.compile` validates the dataset**: every sample must carry the compiled `batchLength`. `LSTM.forward` iterates a fixed `0..<batchLength`, zero-filling short samples and never reading past the window on long ones, neither of which is reported.
- **Pointer-based arithmetic migration**: All layers and optimizers now use `TensorStorage` / `TensorStorage.Pointer` (`UnsafeMutablePointer<Tensor.Scalar>`) and `NumSwiftFlat` pointer APIs instead of `Tensor.Value` (ContiguousArray) arithmetic. See "Pointer-Based Arithmetic" section below.
- **Tensor batching**: `TensorSize` now has a `batchCount` field. `[Tensor].asTensor` packs an array into a single batched tensor. `tensor.batchSlice(b)` extracts a single batch. Axis 3 in `tensor.adding(tensor:axis:)` concatenates along the batch dimension.
- **Optimizer state uses TensorStorage**: Adam, SGD, and RMSProp store momentum/velocity as `[TensorStorage]` instead of `[Tensor.Value]`. SGD returns `forceCopy()` to avoid shared-memory bugs.
- **Device protocol expanded**: `Device` now has pointer-based `conv2d` and `transConv2d` methods accepting `TensorStorage.Pointer` directly.
- **InstanceNormalize fix**: Gradient layout was reversed (gamma|beta vs beta|gamma). Backward now returns `dBeta.concat(dGamma)` to match `weights`; `apply()` uses `depthSlice(0)` for beta, `depthSlice(1)` for gamma. See `InstanceNormalize.swift`.
- **Tensor.flatArray → asArray**: Renamed for consistency with `TensorSize.asArray`. Use `tensor.asArray` for flat `Tensor.Value`.
- **LSTM "exploding gradient" root cause — ragged packed tensors**: `LSTM.weights`/`biases` and the packed gradients from `LSTM.backward` used to concat five differently-shaped groups along axis 2, giving a declared size larger than storage (e.g. bias `[256,1,5]` = 1280 declared vs 1077 stored). Two such tensors added in `GradientAccumulator` match the "along rows" broadcast rule (`rows == 1`) and `broadcastAlongFastPath` iterates by declared size, reading past the buffer every sample. In Release that garbage compounded through heap reuse and showed up as `globalGradientNorm` growing ~10x/step to `inf` while every real parameter gradient stayed < 1. Fix: all five groups are packed flat along axis `-1` (declared size == storage), `apply()` unpacks by offset via `splitPacked`, and the broadcast fast paths `assert` storage == declared size in debug builds. `calculateL2Norm` sums per-tensor `sumOfSquares` instead of concatenating.
- **LSTM gradient bugs (the actual cause of unstable RNN training)**: `LSTMCell.backward` had three errors, found with a finite-difference check (`LSTMNumericalGradientTests`): (1) the input-gate error used `cellError * ia` where the chain rule for `c = f*c_prev + i*g` needs `cellError * ga`; (2) tanh derivatives were computed via `Activation.derivate` on already-activated values (`1 - tanh(tanh(x))^2`) — derivatives of cached post-activation gates must be formed directly (`1 - y^2`, `s(1 - s)`); (3) gate weight gradients used the current `h_t` for the hidden rows instead of `h_{t-1}` (`previousCache.activation`, zeros at t = 0). With these fixed, training descends monotonically with stable weight norms; the old `l2Normalize` in `apply()` (fixed unit-norm steps) only masked the wrong direction by moving fast, and diverged after a few epochs. Any change to `LSTMCell.backward` must keep the FD test passing.
- **`l2Normalize` removed from `LSTM.apply`**: it rescaled each group's optimizer delta to unit norm (~6-14x the nominal Adam step), discarded Adam's scaling, never annealed, and only masked the gradient bugs above. No per-timestep BPTT clipping was added in its place: with correct gradients the LSTM is stable, and genuine explosion on very long sequences is the job of `Optimizer.gradientClip` (rare, spike-only firing).
- **Adam epsilon placement**: `Adam.apply` now computes `mHat / (sqrt(vHat) + eps)` (standard). The previous `sqrt(vHat + eps)` made the step scale-dependent: parameters with small (or heavily clipped) gradients were silently frozen.
- **Per-layer gradient norm inspector**: `Optimizer.gradientNormInspector: (([GradientNormReport]) -> Void)?` is called once per `step()` (all optimizers) with pre-clip L2 norms per layer. Layers adopting `GradientNormInspectable` (currently `LSTM`) report one entry per parameter group (gate) plus an `"(all)"` whole-tensor entry — if `(all)` exceeds the root-sum-square of the groups, storage contains values the layer never applies. Costs nothing while `nil`. See `Optimizers/GradientNormInspection.swift`.

## Overview

Neuron is a Swift-based machine learning framework built from scratch for iOS, macOS, tvOS, and watchOS. It implements neural networks with custom backpropagation, supporting various architectures including CNNs, RNNs, LSTMs, GANs, and more. The framework runs on CPU with C-level optimizations via NumSwift.

## Build & Test Commands

### Building
```bash
swift build
```

### Testing
```bash
# Run all tests (always use CI=true)
CI=true swift test

# Run specific test
CI=true swift test --filter <TestName>
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
- **Tensor**: The fundamental tensor type backed by flat `ContiguousArray<Scalar>` storage with `TensorSize` metadata
- **TensorStorage**: Reference-counted wrapper around `UnsafeMutablePointer<Tensor.Scalar>`. Use `TensorStorage.create(count:)` to allocate and `storage.pointer` for raw access. Prefer constructing tensors with `Tensor(storage:size:)` over `Tensor(Tensor.Value, size:)`.
- **TensorStorage.Pointer**: Typealias for `UnsafeMutablePointer<Tensor.Scalar>`. Use pointer arithmetic (`ptr + offset`) to navigate depth slices and batch slices.
- **Tensor.depthPointer(_:)**: Returns a `TensorStorage.Pointer` to the start of a depth slice without copying. Prefer over `depthSlice(_:)` (which copies into a `Tensor.Value`) in hot paths.
- **Tensor.asArray**: Bridge property returning flat `Tensor.Value` (ContiguousArray); prefer `storage` in hot paths. Formerly `flatArray`.
- **Tensor.value**: Legacy nested `[[[Scalar]]]` view (reconstructed on access); prefer `storage` + `size` in hot paths
- **TensorContext**: Holds backpropagation function for gradient computation
- **TensorSize**: Defines tensor dimensions as `(columns, rows, depth, batchCount)`. `batchCount` defaults to 1. `unitSize` returns `[columns, rows, depth]` without batch.
- Supports automatic gradient calculation via `.gradients(delta:wrt:)` method
- Arithmetic operators overloaded for element-wise and tensor operations
- **Batching**: `[Tensor].asTensor` packs tensors into a single batched tensor. `tensor.batchSlice(b)` extracts one batch. Axis 3 concatenates along batch dimension.

#### Layers (Sources/Neuron/Layers/)
All layers inherit from `BaseLayer` and conform to the `Layer` protocol:
- **BaseLayer**: Base class handling batch processing, device management, weight initialization
- **EncodingType**: Enum defining all layer types for serialization
- Layer categories:
  - Convolutional: `Conv2d`, `TransConv2d`, `MaxPool`, `AvgPool`
  - Dense: `Dense`, `Flatten`, `Reshape`
  - Normalization: `BatchNormalize`, `LayerNormalize`, `InstanceNormalize`
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
- `Device` protocol provides pointer-based `conv2d` and `transConv2d` methods that accept `TensorStorage.Pointer` directly

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

## Design Principles

**Localize changes.** When a feature needs new state, put it on the type that conceptually owns it and thread it through the narrowest path that works. Prefer adding a defaulted parameter to the function that needs the value over adding a stored property to a shared object; prefer a new type over widening an existing one. If a change touches four files, ask whether three of them are being touched only because the state landed in the wrong place.

**`Optimizer` is not a god object.** It has accumulated training-loop policy that isn't conceptually its own — `weightClip`, `gradientClip`, `augmenter`, `ignoreLabelIndex`, `gradientNormInspector`, `learningRateScheduler`. Each was individually reasonable; together they make `Optimizer` the default dumping ground for anything the training loop touches. Before adding another property to `Optimizer` or `BaseOptimizer`:

1. Can it be a parameter on the function that consumes it instead?
2. Does it belong to the loss, the layer, the trainable, or the dataset?
3. If it genuinely has nowhere else to live (usually a construction-order problem — see the `ignoreLabelIndex` entry in `docs/LEARNINGS.md`), add it *and* record the trade-off in `docs/LEARNINGS.md` with the trigger that should move it later.

The same applies to `BaseLayer` and `Tensor`, which are similarly load-bearing. Widening a type that everything depends on is cheap once and expensive forever.

## Code Style (from .cursor/rules/general.mdc)

- Use descriptive variable/function names
- Add comprehensive documentation comments
- Follow Swift naming conventions
- Use `public` for API, `private` for implementation
- Validate input dimensions and provide meaningful errors
- Use NumSwift operations for mathematical computations
- Minimize memory allocations in forward passes
- Create unit tests verifying gradient computations and serialization
- **Never hardcode `Float` or `Float16` when interacting with Tensor**: Always use `Tensor.Scalar` for scalar types, numeric literals (e.g., `Tensor.Scalar(0.0)` not `Float(0.0)`), function signatures, and local variables in any code that touches Tensor. This includes layer parameters, optimizer hyperparameters (learning rate, beta, epsilon), loss function computations, weight initialization, and test assertions. This ensures compatibility with Float16 quantization (when `QUANTIZED_F16` flag is set). See `TensorSIMD.swift` for examples of this pattern.
- **Support both Float and Float16**: Any new math functions on a Tensor should be implemented for both `Float` and `Float16` types. This ensures the framework works correctly regardless of whether quantization is enabled.
- **Prefer pointer-based arithmetic**: Use `TensorStorage.Pointer` and `NumSwiftFlat` APIs instead of `Tensor.Value` array arithmetic. See "Pointer-Based Arithmetic" section below.

## Pointer-Based Arithmetic

The codebase has been migrated to use `TensorStorage` and raw pointers (`TensorStorage.Pointer`) for all hot-path arithmetic, eliminating intermediate `Tensor.Value` (ContiguousArray) allocations.

### Key Patterns

#### Allocating output buffers
```swift
// OLD: var result = Tensor.Value(repeating: 0, count: n)
// NEW:
let result = TensorStorage.create(count: n)
```

#### Accessing depth slices without copying
```swift
// OLD: let slice = tensor.depthSlice(d)  // copies into Tensor.Value
// NEW:
let ptr = tensor.storage.pointer + d * sliceSize  // zero-copy pointer offset
// or:
let ptr = tensor.depthPointer(d)
```

#### Element-wise arithmetic via NumSwiftFlat
```swift
// OLD: let scaled = (normalized * gamma[i]) + beta[i]
// NEW:
NumSwiftFlat.mul(normPtr, scalar: gamma[i], result: outPtr, count: sliceSize)
NumSwiftFlat.add(outPtr, scalar: beta[i], result: outPtr, count: sliceSize)
```

Available `NumSwiftFlat` pointer operations: `add`, `sub`, `mul`, `div`, `sqrt`, `sum` (scalar and pointer-pointer variants).

#### Constructing Tensors from TensorStorage
```swift
// OLD: Tensor(arrayValue, size: size)
// NEW:
Tensor(storage: tensorStorage, size: size)
Tensor(storage: tensorStorage, size: size, context: tensorContext)
```

#### Reusable scratch buffers
When looping over depth slices, allocate temp buffers once outside the loop:
```swift
let tmpA = TensorStorage.create(count: sliceSize)
let tmpB = TensorStorage.create(count: sliceSize)
for d in 0..<depth {
  // reuse tmpA, tmpB for intermediate calculations each iteration
}
```

#### Pointer copy
```swift
dstPtr.update(from: srcPtr, count: sliceSize)
```

#### SGD shared-memory pitfall
When optimizer state (velocity) is returned as a Tensor, use `forceCopy()` to avoid the tensor sharing mutable memory with the optimizer:
```swift
return (Tensor(storage: v[i].forceCopy(), size: gradient.size), ...)
```

### Migration Checklist for New Layers
1. Replace `Tensor.Value(repeating: 0, count:)` with `TensorStorage.create(count:)`
2. Replace `tensor.depthSlice(d)` with `tensor.storage.pointer + d * sliceSize` or `tensor.depthPointer(d)`
3. Replace `Tensor.Value` arithmetic operators (`+`, `-`, `*`, `/`) with `NumSwiftFlat` pointer functions
4. Replace `Tensor(array, size:)` with `Tensor(storage:, size:)`
5. Store optimizer state as `[TensorStorage]` instead of `[Tensor.Value]`

## Dependencies

- **NumSwift**: C-optimized numeric operations (SIMD, BLAS-like functions)
- **Logger**: Logging framework for debugging
- **swift-numerics**: Apple's numerics library

## Branch Strategy

- `main`: Stable production branch
- `develop`: Development branch for integration
- Feature branches: Branch off `develop`, PR into `develop`
- **All PRs must target `develop` as the base branch**, not `main`. The `main` branch is only updated via merges from `develop`.
- Automated tests must pass before PR merge

### Commit history

- **Squash-merge only.** Merge commits are disabled on the repo; every PR lands as one commit on its base branch.
- **The PR description becomes the commit message.** The repo is set to `squash_merge_commit_title: PR_TITLE` / `squash_merge_commit_message: PR_BODY`, so whatever is in the PR body is what `git log` shows forever. Write it as release notes, not as a work log. (This was previously `COMMIT_MESSAGES`, which concatenated every commit in the PR — and, because feature squashes carried those bodies too, re-concatenated them into each release. The `develop` → `main` squash for #183 ended up 1,926 lines long and reached back to PR #39.)
- **`.github/pull_request_template.md` deliberately contains no HTML comments.** GitHub does not strip `<!-- -->` when building the squash body, so instructional comments would land verbatim in `git log`. Keep the template short and comment-free.
- **Pull with rebase.** `pull.rebase=true` is set locally in this repo; set it on any other machine you work from. Without it, pulling `develop` produces `Merge branch 'develop' of github.com:... into develop` commits that clutter the release PR.
- **After each release, back-merge `main` into `develop`.** The squash commit on `main` is not an ancestor of `develop`, so without this the next release PR computes the wrong merge base and re-lists already-shipped commits.

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
- Tensors are logically 3D with an optional batch dimension: `[columns, rows, depth]` + `batchCount`
- `TensorSize` has `batchCount` (defaults to 1). Use `size.unitSize` for `[columns, rows, depth]` without batch.
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

### Reference Documents
- **docs/AGENT_REFERENCE.md**: Condensed reference for AI agents (architecture, optimizer gradient layout, InstanceNormalize fix, common pitfalls)
- **docs/LEARNINGS.md**: Deliberate trade-offs and deferred decisions — where a constraint forced a compromise and what should trigger a revisit. Read before "fixing" something that looks misplaced; it may be a recorded compromise with a reason.
- **docs/GPU_ARCHITECTURE_LEARNINGS.md**: Why the Metal GPU path benchmarked ~5x slower than CPU on MNIST, and what a future attempt should do differently. The implementation it describes has been removed; the measurements and root-cause analysis are what's kept.

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
4. Shape mismatches: Remember all tensors are 3D internally (+ optional batch dimension)
5. Not setting `isTraining = true` before training (affects Dropout, BatchNorm)
6. Accessing `learningRate` when `decayFunction` is set (use property, not field)
7. Not calling `optimizer.step()` after `apply()` (needed for decay function updates)
8. **Gradient/weights layout mismatch**: Layers with multiple params (e.g. gamma, beta) must return gradients in the same order as `weights`. InstanceNormalize and LayerNormalize use `beta | gamma`; gradients must match or Adam weight decay corrupts training. See `InstanceNormalize.swift` and `AGENT_REFERENCE.md`.
9. **Using `Tensor.Value` arithmetic in hot paths**: Prefer `NumSwiftFlat` pointer APIs to avoid intermediate array allocations. See "Pointer-Based Arithmetic" section.
10. **Shared-memory bugs with optimizer state**: When returning optimizer state (e.g., SGD velocity) as a Tensor, use `TensorStorage.forceCopy()` to avoid the tensor mutating optimizer internals.
11. **Tensor self-assignment creates reference cycles**: Tensor arithmetic operators (`+`, `-`, `*`, `/`) build autograd computation graphs. Writing `gamma = gamma - gradients` creates a reference cycle because the result's graph holds a reference to the old `gamma`, which is now the same variable as the new `gamma`. Instead, construct a fresh Tensor from the result's storage: `gamma = Tensor(storage: (gamma - gradients).storage, size: gamma.size)`. Alternatively, perform the arithmetic at the `TensorStorage` level (which has no autograd): `gamma = Tensor(storage: gamma.storage - gradients.storage, size: gamma.size)`. This applies anywhere a Tensor property is updated via arithmetic that references itself.
12. **`gradientClip` cannot bound updates in front of Adam**: global-norm clipping scales every gradient by `c` before the moment update, and Adam's `mHat/sqrt(vHat)` cancels `c`. If the clip fires every step it does nothing to the weight trajectory (and pre-fix, the eps floor made it *freeze* the non-exploding parameters). Use it as rare spike protection only (e.g. ~5, not 1.0), and use `gradientNormInspector` to find which parameter group is actually large before reaching for it.
13. **Never build a tensor whose declared size differs from its storage**: `concat(axis: 2)` of differently-shaped tensors appends storage but reports the first tensor's rows/columns, so `rows*columns*depth > storage.count`. Broadcast fast paths and `debugDescription` walk the declared size and read out of bounds (silently in Release). Pack heterogeneous groups flat with `concat(axis: -1)` and unpack by offset, as `LSTM` does. `Layer.swift`, `LayerGroup`, and `ResNet` still concat sub-layer weights along axis 2 for their `weights` views — audit before doing arithmetic on those.

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

#### Allocation Minimization
- Use `TensorStorage.create(count:)` instead of `Tensor.Value(repeating:count:)` for output buffers.
- Use `NumSwiftFlat` pointer APIs instead of `Tensor.Value` operator overloads to avoid intermediate array allocations.
- Access depth slices via pointer offset (`storage.pointer + d * sliceSize`) instead of `depthSlice(d)` which copies.
- Allocate reusable scratch `TensorStorage` buffers outside loops and reuse them each iteration.
- Pre-allocate when output size is known (especially in recurrent loops).

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

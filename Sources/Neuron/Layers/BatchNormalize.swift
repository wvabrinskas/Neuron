//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Performs a normalization of the inputs based on the batch.
///
/// ## Thread Synchronization for Batch Processing
///
/// This layer uses a sophisticated thread synchronization pattern for concurrent batch processing:
///
/// ```
/// Multiple Threads Processing Batch Items
/// ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
/// │ Thread 1 │  │ Thread 2 │  │ Thread 3 │  │ Thread N │
/// │ Tensor A │  │ Tensor B │  │ Tensor C │  │ Tensor D │
/// └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘
///      │             │             │             │
///      └─────────────┼─────────────┼─────────────┘
///                    │             │
///                    ▼             ▼
///            ┌─────────────────────────────┐
///            │    NSCondition.lock()       │
///            │  (sync point for batch)     │
///            └─────────────────────────────┘
///                          │
///                          ▼
///            ┌─────────────────────────────┐
///            │   Each thread executes:     │
///            │   iterations += 1           │ ◄─── @Atomic property
///            │   calculateWelfordVariance  │
///            └─────────────────────────────┘
///                          │
///                          ▼
///            ┌─────────────────────────────┐
///            │   updateLock.with { ... }   │ ◄─── NSLock protection
///            │   - Update means[@Atomic]   │
///            │   - Update m2s[@Atomic]     │
///            │   - Welford variance calc   │
///            └─────────────────────────────┘
///                          │
///                          ▼
///            ┌─────────────────────────────┐
///            │  if iterations == batchSize │
///            │    condition.broadcast()    │ ◄─── Signal all waiting
///            │  else                       │
///            │    condition.wait()         │ ◄─── Wait for completion
///            └─────────────────────────────┘
///                          │
///                          ▼
///            ┌─────────────────────────────┐
///            │   condition.unlock()        │
///            │   All threads synchronized  │
///            └─────────────────────────────┘
///                          │
///                          ▼
///            ┌─────────────────────────────┐
///            │  Individual normalize3D()   │
///            │  - Use cached statistics    │
///            │  - Apply gamma/beta scaling │
///            │  - Store in ThreadStorage   │ ◄─── Thread-local cache
///            └─────────────────────────────┘
/// ```
///
/// **Key Synchronization Points:**
/// - `NSCondition` (`condition`): Ensures all threads wait until the entire batch has computed statistics
/// - `NSLock` (`updateLock`): Protects shared state updates during Welford variance calculation
/// - `@Atomic` properties (`iterations`, `means`, `m2s`): Thread-safe counters and accumulators
/// - `ThreadStorage` (`cachedNormalizations`): Thread-local storage for normalization values
///
/// **Process Flow:**
/// 1. Threads process tensors concurrently from BaseOptimizer
/// 2. Each thread increments atomic `iterations` counter
/// 3. Welford variance calculation updates shared `means` and `m2s` under lock
/// 4. Threads wait at condition barrier until all batch items processed
/// 5. Once synchronized, each thread normalizes using computed batch statistics
///
/// This two-phase approach (statistics collection + individual normalization) ensures correct
/// batch normalization while maximizing concurrency.
public final class BatchNormalize: BaseThreadBatchingLayer {
  /// Learnable scale parameters applied after normalization, one per depth slice.
  public var gamma: [Tensor.Scalar] = []
  /// Learnable shift parameters applied after normalization, one per depth slice.
  public var beta: [Tensor.Scalar] = []
  /// Per-depth-slice moving mean, stored as flat arrays
  public var movingMean: Tensor = .init()
  /// Per-depth-slice moving variance, stored as flat arrays
  public var movingVariance: Tensor = .init()
  
  /// The momentum factor used to update moving mean and variance during training.
  public let momentum: Tensor.Scalar
  /// A combined tensor of beta, gamma, moving mean, and moving variance values, used for display purposes only.
  /// Setting this property has no effect.
  public override var weights: Tensor {
    // only used for print purposes.
    get {
      var beta = beta
      beta.append(contentsOf: gamma)
      beta.append(contentsOf: movingMean.storage)
      beta.append(contentsOf: movingVariance.storage)
      
      // size is not needed here as gradients aren't applied from the optimizer
      return Tensor(beta)
    }
    set {}
  }
  
  /// Indicates whether the layer should accumulate inputs into batches, returning `true` during training.
  public override var shouldPerformBatching: Bool {
    isTraining
  }
  
  private let e: Tensor.Scalar = 1e-5 //this is a standard smoothing term
  private var dGamma: [Tensor.Scalar] = []
  private var dBeta: [Tensor.Scalar] = []
  internal let welfordVariance = WelfordVariance()
  
  private struct Normalization {
    let normalized: TensorStorage
    let std: TensorStorage
  }
  
  /// Creates a batch-normalization layer.
  ///
  /// - Parameters:
  ///   - gamma: Optional per-channel scale parameters.
  ///   - beta: Optional per-channel bias parameters.
  ///   - momentum: Moving-average momentum for inference statistics.
  ///   - movingMean: Optional preloaded moving mean tensor.
  ///   - movingVariance: Optional preloaded moving variance tensor.
  ///   - inputSize: Optional input tensor shape.
  public init(gamma: [Tensor.Scalar] = [],
              beta: [Tensor.Scalar] = [],
              momentum: Tensor.Scalar = 0.99,
              movingMean: Tensor = .init(),
              movingVariance: Tensor = .init(),
              inputSize: TensorSize? = nil,
              linkId: String = UUID().uuidString) {
    self.gamma = gamma
    self.beta = beta
    self.movingMean = movingMean
    self.movingVariance = movingVariance
    self.momentum = momentum
    
    super.init(inputSize: inputSize,
               linkId: linkId,
               encodingType: .batchNormalize)
    
    self.usesOptimizer = false
    setupTrainables()
    resetDeltas()
    
    if let inputSize {
      welfordVariance.setInputSize(inputSize)
    }
  }
  
  /// Coding keys used to encode and decode the batch-normalization layer's persistent properties.
  public enum CodingKeys: String, CodingKey {
    case gamma, beta, momentum, movingMean, movingVariance, inputSize, linkId
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let gamma = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .gamma) ?? []
    let beta = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .beta) ?? []
    let momentum = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .momentum) ?? 0.99
    
    let inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    
    //backwards compatibility for mean
    let movingMean: Tensor = if let tensorMean = try? container.decodeIfPresent(Tensor.self, forKey: .movingMean) {
      tensorMean
    } else if let tensorDataMean = try? container.decodeIfPresent(Tensor.Data.self, forKey: .movingMean) {
      Tensor(tensorDataMean)
    } else if let tensorValueMean = try? container.decodeIfPresent([Tensor.Scalar].self, forKey: .movingMean) {
      Tensor(Tensor.Value(tensorValueMean.flatMap { Tensor.Value(repeating: $0, count: inputSize.rows * inputSize.columns) }), size: inputSize)
    } else {
      .init()
    }
    
    //backwards compatibility for variance
    let movingVar: Tensor = if let tensorVar = try? container.decodeIfPresent(Tensor.self, forKey: .movingVariance) {
      tensorVar
    } else if let tensorDataVar = try? container.decodeIfPresent(Tensor.Data.self, forKey: .movingVariance) {
      Tensor(tensorDataVar)
    } else if let tensorValueVar = try? container.decodeIfPresent([Tensor.Scalar].self, forKey: .movingVariance) {
      Tensor(Tensor.Value(tensorValueVar.flatMap { Tensor.Value(repeating: $0, count: inputSize.rows * inputSize.columns) }), size: inputSize)
    } else {
      .init()
    }
    
    if movingMean.isEmpty || movingVar.isEmpty {
      fatalError("Couldn't decode movingMean or movingVariance")
    }
        
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    
    self.init(gamma: gamma,
              beta: beta,
              momentum: momentum,
              movingMean: movingMean,
              movingVariance: movingVar,
              linkId: linkId)
    
    self.inputSize = inputSize
    self.outputSize = inputSize
  }
  
  /// Encodes batch-normalization parameters and running statistics.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(momentum, forKey: .momentum)
    try container.encode(movingMean, forKey: .movingMean)
    try container.encode(movingVariance, forKey: .movingVariance)
    try container.encode(linkId, forKey: .linkId)
  }
  
  // actual forward pass happens here from the super class
  /// Updates running batch statistics for one tensor during synchronized batching.
  ///
  /// - Parameters:
  ///   - tensor: Tensor participating in the current batch.
  ///   - context: Batch/thread metadata used for synchronization.
  public override func performThreadBatchingForwardPass(tensor: Tensor, context: NetworkContext) {
    updateWelford(inputs: tensor, context: context)
  }
  
  /// Applies batch normalization to an input tensor.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Batch/thread metadata used for synchronized statistics.
  /// - Returns: Normalized tensor with attached backpropagation context.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let forward = normalize(inputs: tensor, context: context)
    let normalizations = forward.normalized
    
    if iterations.load(ordering: .relaxed) == 0 && context.totalInBatch == 1 {
      updateWelford(inputs: tensor, context: context)
    }
    
    let tensorContext = TensorContext { inputs, gradient, wrt in
      let backward = self.backward(inputs: inputs,
                                   gradient: gradient,
                                   context: context,
                                   normalizations: normalizations)
      
      backward.label = "BatchNorm"
      return (backward, Tensor(), Tensor())
    }
    
    let out = Tensor(storage: forward.output, size: tensor.size, context: tensorContext)
    
    out.setGraph(tensor)
    
    return super.forward(tensor: out, context: context)
  }
  
  /// Applies gradients to `gamma` and `beta`, then resets batch accumulators.
  ///
  /// - Parameters:
  ///   - gradients: Unused optimizer gradient tuple for this layer type.
  ///   - learningRate: Learning rate used to scale `gamma`/`beta` updates.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    super.apply(gradients: gradients, learningRate: learningRate)
    
    let avgDGamma = dGamma / welfordVariance.iterations.asTensorScalar
    let avgDBeta = dBeta / welfordVariance.iterations.asTensorScalar
    
    gamma = gamma - (avgDGamma * learningRate)
    beta = beta - (avgDBeta * learningRate)
    
    resetDeltas()
    welfordVariance.reset()
  }
  
  /// Rebuilds internal trainables when input shape changes.
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    welfordVariance.setInputSize(inputSize)
    setupTrainables()
    resetDeltas()
  }
  
  private func setupTrainables() {
    let inputDim = inputSize.depth
    
    if gamma.isEmpty {
      gamma = [Tensor.Scalar](repeating: 1, count: inputDim)
    }
    
    if beta.isEmpty {
      beta = [Tensor.Scalar](repeating: 0, count: inputDim)
    }
    
    if movingMean.isEmpty {
      movingMean = .fillWith(value: 0, size: inputSize)
    }
    
    if movingVariance.isEmpty {
      movingVariance = .fillWith(value: 1, size: inputSize)
    }
  }
  
  private func resetDeltas() {
    let inputDim = inputSize.depth
    dGamma = [Tensor.Scalar](repeating: 0, count: inputDim)
    dBeta = [Tensor.Scalar](repeating: 0, count: inputDim)
  }
  
  private func updateWelford(inputs: Tensor, context: NetworkContext) {
    guard isTraining else { return }
    
    updateLock.with {
      welfordVariance.update(inputs)
      
      if welfordVariance.iterations > batchSize {
        fatalError()
      }
    }
  }
  
  private func normalize(inputs: Tensor, context: NetworkContext) -> (output: TensorStorage, normalized: [Normalization]) {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let outStorage = TensorStorage.create(count: inputs.storage.count)
    var normalizedInputs: [Normalization] = []

    // Reusable temp buffers (single-depth-slice sized)
    let tmpA = TensorStorage.create(count: sliceSize)
    let tmpB = TensorStorage.create(count: sliceSize)

    for i in 0..<depth {
      let inputPtr  = inputs.storage.pointer + i * sliceSize
      let outPtr    = outStorage.pointer + i * sliceSize

      if isTraining {
        updateLock.with {
          let (meanStorage, stdStorage, normStorage) =
            calculateWelfordVariables(inputPtr: inputPtr, sliceSize: sliceSize,
                                      index: i, batchSize: context.totalInBatch,
                                      varScratch: tmpA)
          normalizedInputs.append(Normalization(normalized: normStorage, std: stdStorage))

          // scaled = normalized * gamma[i] + beta[i]
          NumSwiftFlat.mul(normStorage.pointer, scalar: gamma[i], result: outPtr, count: sliceSize)
          NumSwiftFlat.add(outPtr, scalar: beta[i], result: outPtr, count: sliceSize)

          // Update moving stats: movingX = movingX * momentum + x * (1 - momentum)
          // tmpA holds variance on return from calculateWelfordVariables (see docstring).
          // Update movingVariance first, before tmpA is reused for the mean blending.
          let meanPtr = meanStorage.pointer
          let mmPtr = movingMean.storage.pointer + i * sliceSize
          let mvPtr = movingVariance.storage.pointer + i * sliceSize

          // movingVariance update: tmpA holds variance from calculateWelfordVariables
          // tmpB = variance * (1 - momentum)
          NumSwiftFlat.mul(tmpA.pointer, scalar: 1 - momentum, result: tmpB.pointer, count: sliceSize)
          // tmpA = movingVariance[i] * momentum  (overwrites variance — no longer needed)
          NumSwiftFlat.mul(mvPtr, scalar: momentum, result: tmpA.pointer, count: sliceSize)
          // movingVariance[i] = tmpA + tmpB
          NumSwiftFlat.add(tmpA.pointer, tmpB.pointer, result: mvPtr, count: sliceSize)

          // movingMean update
          // tmpA = movingMean[i] * momentum
          NumSwiftFlat.mul(mmPtr, scalar: momentum, result: tmpA.pointer, count: sliceSize)
          // tmpB = mean * (1 - momentum)
          NumSwiftFlat.mul(meanPtr, scalar: 1 - momentum, result: tmpB.pointer, count: sliceSize)
          // movingMean[i] = tmpA + tmpB
          NumSwiftFlat.add(tmpA.pointer, tmpB.pointer, result: mmPtr, count: sliceSize)
        }
      } else {
        let mmPtr = movingMean.storage.pointer + i * sliceSize
        let mvPtr = movingVariance.storage.pointer + i * sliceSize

        // std = sqrt(mv + e)
        NumSwiftFlat.add(mvPtr, scalar: e, result: tmpA.pointer, count: sliceSize)
        NumSwiftFlat.sqrt(tmpA.pointer, result: tmpA.pointer, count: sliceSize)

        // normalized = (input - mm) / std
        NumSwiftFlat.sub(inputPtr, mmPtr, result: tmpB.pointer, count: sliceSize)
        NumSwiftFlat.div(tmpB.pointer, tmpA.pointer, result: tmpB.pointer, count: sliceSize)

        // scaled = normalized * gamma[i] + beta[i]
        NumSwiftFlat.mul(tmpB.pointer, scalar: gamma[i], result: outPtr, count: sliceSize)
        NumSwiftFlat.add(outPtr, scalar: beta[i], result: outPtr, count: sliceSize)
      }
    }

    return (outStorage, normalizedInputs)
  }

  /// Returns (mean, std, normalized) as `TensorStorage` instances for depth slice `index`.
  /// `varScratch` is a caller-provided scratch buffer of size `sliceSize`. On return it holds
  /// the per-element variance (`m2 / batchSize`) for depth slice `index`. Callers may read
  /// this value after the call; it is guaranteed to be valid until `varScratch` is next passed
  /// to this function or written by the caller.
  private func calculateWelfordVariables(inputPtr: TensorStorage.Pointer,
                                         sliceSize: Int,
                                         index: Int,
                                         batchSize: Int,
                                         varScratch: TensorStorage) -> (mean: TensorStorage,
                                                                         std: TensorStorage,
                                                                         normalized: TensorStorage) {
    // Copy Welford mean/m2 slices (ContiguousArray) into TensorStorage for pointer access
    let meanStorage = TensorStorage(welfordVariance.means[index])
    let m2Storage   = TensorStorage(welfordVariance.m2s[index])

    // variance = m2 / batchSize  (written into caller-supplied scratch)
    NumSwiftFlat.div(m2Storage.pointer, scalar: Tensor.Scalar(batchSize),
                      result: varScratch.pointer, count: sliceSize)

    // std = sqrt(variance + e)
    let stdStorage = TensorStorage.create(count: sliceSize)
    NumSwiftFlat.add(varScratch.pointer, scalar: e, result: stdStorage.pointer, count: sliceSize)
    NumSwiftFlat.sqrt(stdStorage.pointer, result: stdStorage.pointer, count: sliceSize)

    // normalized = (input - mean) / std
    let normStorage = TensorStorage.create(count: sliceSize)
    NumSwiftFlat.sub(inputPtr, meanStorage.pointer, result: normStorage.pointer, count: sliceSize)
    NumSwiftFlat.div(normStorage.pointer, stdStorage.pointer, result: normStorage.pointer, count: sliceSize)

    return (meanStorage, stdStorage, normStorage)
  }

  private func backward(inputs: Tensor,
                        gradient: Tensor,
                        context: NetworkContext,
                        normalizations: [Normalization]) -> Tensor {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let outStorage = TensorStorage.create(count: inputs.storage.count)

    let N = Tensor.Scalar(context.totalInBatch)

    // Reusable scratch buffers
    let tmpA = TensorStorage.create(count: sliceSize)
    let dxNorm     = TensorStorage.create(count: sliceSize)
    let dxNormXNorm = TensorStorage.create(count: sliceSize)
    let term       = TensorStorage.create(count: sliceSize)
    let invNStd    = TensorStorage.create(count: sliceSize)

    for i in 0..<depth {
      let gradPtr  = gradient.storage.pointer + i * sliceSize
      let outPtr   = outStorage.pointer + i * sliceSize

      let normStorage: TensorStorage
      let stdStorage: TensorStorage

      if let norm = normalizations[safe: i] {
        normStorage = norm.normalized
        stdStorage  = norm.std
      } else {
        let inputPtr = inputs.storage.pointer + i * sliceSize
        let result = calculateWelfordVariables(inputPtr: inputPtr, sliceSize: sliceSize,
                                               index: i, batchSize: context.totalInBatch,
                                               varScratch: tmpA)
        normStorage = result.normalized
        stdStorage  = result.std
      }

      updateLock.with {
        // dGamma[i] += sum(grad * normalized)
        NumSwiftFlat.mul(gradPtr, normStorage.pointer, result: tmpA.pointer, count: sliceSize)
        dGamma[i] += NumSwiftFlat.sum(tmpA.pointer, count: sliceSize)
        // dBeta[i] += sum(grad)
        dBeta[i] += NumSwiftFlat.sum(gradPtr, count: sliceSize)
      }

      // dxNorm = grad * gamma[i]
      NumSwiftFlat.mul(gradPtr, scalar: gamma[i], result: dxNorm.pointer, count: sliceSize)

      // dxNormSum = sum(dxNorm)
      let dxNormSum = NumSwiftFlat.sum(dxNorm.pointer, count: sliceSize)

      // dxNormTimesNormSum = sum(dxNorm * normalized)
      NumSwiftFlat.mul(dxNorm.pointer, normStorage.pointer, result: dxNormXNorm.pointer, count: sliceSize)
      let dxNormTimesNormSum = NumSwiftFlat.sum(dxNormXNorm.pointer, count: sliceSize)

      // term = N * dxNorm - dxNormSum - normalized * dxNormTimesNormSum
      NumSwiftFlat.mul(dxNorm.pointer, scalar: N, result: term.pointer, count: sliceSize)
      NumSwiftFlat.sub(term.pointer, scalar: dxNormSum, result: term.pointer, count: sliceSize)
      // tmpA = normalized * dxNormTimesNormSum
      NumSwiftFlat.mul(normStorage.pointer, scalar: dxNormTimesNormSum, result: tmpA.pointer, count: sliceSize)
      NumSwiftFlat.sub(term.pointer, tmpA.pointer, result: term.pointer, count: sliceSize)

      // invNStd = (1 / N) / std
      NumSwiftFlat.div(scalar: 1 / N, stdStorage.pointer, result: invNStd.pointer, count: sliceSize)

      // dx = invNStd * term
      NumSwiftFlat.mul(invNStd.pointer, term.pointer, result: outPtr, count: sliceSize)
    }

    return Tensor(storage: outStorage, size: inputs.size)
  }
  
}


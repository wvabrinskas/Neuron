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
  
  private struct NormalizationFlat {
    let normalized: Tensor.Value
    let std: Tensor.Value
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
    
    let out = Tensor(forward.output, size: tensor.size, context: tensorContext)
    
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
  
  private func normalize(inputs: Tensor, context: NetworkContext) -> (output: Tensor.Value, normalized: [NormalizationFlat]) {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    var outStorage = Tensor.Value(repeating: 0, count: inputs.storage.count)
    var normalizedInputs: [NormalizationFlat] = []
    
    for i in 0..<depth {
      let inputSlice = inputs.depthSlice(i)
      let outOffset = i * sliceSize
      
      if isTraining {
        updateLock.with {
          let (mean, variance, std, normalized) = calculateWelfordVariables(inputs: inputSlice, index: i,
                                                                            batchSize: context.totalInBatch)
          normalizedInputs.append(NormalizationFlat(normalized: normalized, std: std))
          
          // gamma[i] * normalized + beta[i]
          let scaled = (normalized * gamma[i]) + beta[i]
          
          // Update moving stats
          movingMean.setDepthSlice(i, (movingMean.depthSlice(i) * momentum) + (mean * (1 - momentum)))
          movingVariance.setDepthSlice(i, (movingVariance.depthSlice(i) * momentum) + (variance * (1 - momentum)))
          
          for j in 0..<sliceSize { outStorage[outOffset + j] = scaled[j] }
        }
      } else {
        let mm = movingMean.depthSlice(i)
        let mv = movingVariance.depthSlice(i)
        // std = sqrt(mv + e)
        let std = NumSwiftFlat.sqrt(mv + e)
        // normalized = (input - mm) / std
        let normalized = (inputSlice - mm) / std
        // output = gamma[i] * normalized + beta[i]
        let scaled = (normalized * gamma[i]) + beta[i]
        
        for j in 0..<sliceSize { outStorage[outOffset + j] = scaled[j] }
      }
    }
    
    return (outStorage, normalizedInputs)
  }
  
  private func calculateWelfordVariables(inputs: Tensor.Value,
                                         index: Int,
                                         batchSize: Int) -> (mean: Tensor.Value,
                                                             variance: Tensor.Value,
                                                             std: Tensor.Value,
                                                             out: Tensor.Value) {
    let mean = welfordVariance.means[index]
    let variance = welfordVariance.m2s[index] / Tensor.Scalar(batchSize)
    
    let std = NumSwiftFlat.sqrt(variance + e)
    let normalized = (inputs - mean) / std
    
    return (mean, variance, std, normalized)
  }
  
  private func backward(inputs: Tensor,
                        gradient: Tensor,
                        context: NetworkContext,
                        normalizations: [NormalizationFlat]) -> Tensor {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    var outStorage = Tensor.Value(repeating: 0, count: inputs.storage.count)
    
    for i in 0..<depth {
      let N = Tensor.Scalar(context.totalInBatch)
      let gradSlice = gradient.depthSlice(i)
      
      var normalized: Tensor.Value
      var std: Tensor.Value
      
      if let norm = normalizations[safe: i] {
        normalized = norm.normalized
        std = norm.std
      } else {
        let inputSlice = inputs.depthSlice(i)
        let (_, _, nStd, nNormalized) = calculateWelfordVariables(inputs: inputSlice, index: i, batchSize: context.totalInBatch)
        normalized = nNormalized
        std = nStd
      }
      
      updateLock.with {
        dGamma[i] +=  (gradSlice * normalized).sum
        dBeta[i] += gradSlice.sum
      }
      
      // dxNorm = gradient[i] * gamma[i]
      let dxNorm = gradSlice * gamma[i]
      
      // dx = 1 / N / std * (N * dxNorm - dxNorm.sum - normalized * (dxNorm * normalized).sum)
      let dxNormSum = dxNorm.sum
      let dxNormTimesNorm = dxNorm * normalized
      let dxNormTimesNormSum = dxNormTimesNorm.sum
      
      let term1 = dxNorm * N
      let term2 = term1 - dxNormSum
      let term3 = term2 - (normalized * dxNormTimesNormSum)
      
      let invNStd = (Tensor.Value(repeating: 1, count: sliceSize) / N) / std
      
      let dx = invNStd * term3
      
      let outOffset = i * sliceSize
      for j in 0..<sliceSize { outStorage[outOffset + j] = dx[j] }
    }
    
    return Tensor(outStorage, size: inputs.size)
  }
  
}


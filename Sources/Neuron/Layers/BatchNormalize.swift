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
  public var gamma: [Tensor.Scalar] = []
  public var beta: [Tensor.Scalar] = []
  /// Per-depth-slice moving mean, stored as flat arrays
  public var movingMeanSlices: [ContiguousArray<Tensor.Scalar>] = []
  /// Per-depth-slice moving variance, stored as flat arrays
  public var movingVarianceSlices: [ContiguousArray<Tensor.Scalar>] = []
  
  public let momentum: Tensor.Scalar
  public override var weights: Tensor {
    // only used for print purposes.
    get {
      var beta = beta
      beta.append(contentsOf: gamma)
      return Tensor(beta)
    }
    set {}
  }
  
  public override var shouldPerformBatching: Bool {
    isTraining
  }
  
  private let e: Tensor.Scalar = 1e-5 //this is a standard smoothing term
  private var dGamma: [Tensor.Scalar] = []
  private var dBeta: [Tensor.Scalar] = []
  internal let welfordVariance = WelfordVariance()

  private struct NormalizationFlat {
    let normalized: ContiguousArray<Tensor.Scalar>
    let std: ContiguousArray<Tensor.Scalar>
  }
  
  /// Default initializer for Batch Normalize layer
  public init(gamma: [Tensor.Scalar] = [],
              beta: [Tensor.Scalar] = [],
              momentum: Tensor.Scalar = 0.99,
              movingMean: Tensor.Data = [],
              movingVariance: Tensor.Data = [],
              inputSize: TensorSize? = nil) {
    self.gamma = gamma
    self.beta = beta
    // Convert nested arrays to flat slices
    self.movingMeanSlices = movingMean.map { ContiguousArray($0.flatMap { $0 }) }
    self.movingVarianceSlices = movingVariance.map { ContiguousArray($0.flatMap { $0 }) }
    self.momentum = momentum
    
    super.init(inputSize: inputSize,
               encodingType: .batchNormalize)
    
    setupTrainables()
    resetDeltas()
    
    if let inputSize {
      welfordVariance.setInputSize(inputSize)
    }
  }
  
  public enum CodingKeys: String, CodingKey {
    case gamma, beta, momentum, movingMean, movingVariance, inputSize
  }

  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let movingMean = try container.decodeIfPresent(Tensor.Data.self, forKey: .movingMean) ?? []
    let movingVar = try container.decodeIfPresent(Tensor.Data.self, forKey: .movingVariance) ?? []
    let gamma = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .gamma) ?? []
    let beta = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .beta) ?? []
    let momentum = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .momentum) ?? 0.99

    self.init(gamma: gamma,
              beta: beta,
              momentum: momentum,
              movingMean: movingMean,
              movingVariance: movingVar)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(momentum, forKey: .momentum)
    
    // Convert flat slices back to nested arrays for Codable compatibility
    let cols = inputSize.columns
    let mmNested: Tensor.Data = movingMeanSlices.map { slice in
      stride(from: 0, to: slice.count, by: cols).map { Array(slice[$0..<min($0 + cols, slice.count)]) }
    }
    let mvNested: Tensor.Data = movingVarianceSlices.map { slice in
      stride(from: 0, to: slice.count, by: cols).map { Array(slice[$0..<min($0 + cols, slice.count)]) }
    }
    try container.encode(mmNested, forKey: .movingMean)
    try container.encode(mvNested, forKey: .movingVariance)
  }
  
  // actual forward pass happens here from the super class
  public override func performThreadBatchingForwardPass(tensor: Tensor, context: NetworkContext) {
    calculateWelfordVariance(inputs: tensor, context: context)
  }

  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let forward = normalize3DFlat(inputs: tensor, context: context)
    let normalizations = forward.normalized

    if iterations.load(ordering: .relaxed) == 0 && context.totalInBatch == 1 {
      calculateWelfordVariance(inputs: tensor, context: context)
    }
    
    let tensorContext = TensorContext { inputs, gradient, wrt in
      let backward = self.backwardFlat(inputs: inputs,
                                        gradient: gradient,
                                        context: context,
                                        normalizations: normalizations)
      
      backward.label = "BatchNorm"
      return (backward, Tensor(), Tensor())
    }
    
    let out = Tensor(forward.output, size: tensor.size, context: tensorContext)
    
    out.setGraph(tensor)
    out.label = "BatchNorm"
    
    return out
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    super.apply(gradients: gradients, learningRate: learningRate)
        
    let avgDGamma = dGamma / welfordVariance.iterations.asTensorScalar
    let avgDBeta = dBeta / welfordVariance.iterations.asTensorScalar
    
    gamma = gamma - (avgDGamma * learningRate)
    beta = beta - (avgDBeta * learningRate)
    
    resetDeltas()
    welfordVariance.reset()
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    welfordVariance.setInputSize(inputSize)
    setupTrainables()
    resetDeltas()
  }
  
  private func setupTrainables() {
    let inputDim = inputSize.depth
    let sliceSize = inputSize.rows * inputSize.columns
    
    if gamma.isEmpty {
      gamma = [Tensor.Scalar](repeating: 1, count: inputDim)
    }
    
    if beta.isEmpty {
      beta = [Tensor.Scalar](repeating: 0, count: inputDim)
    }
  
    if movingMeanSlices.isEmpty {
      movingMeanSlices = [ContiguousArray<Tensor.Scalar>](repeating: ContiguousArray(repeating: 0, count: sliceSize), count: inputDim)
    }
    
    if movingVarianceSlices.isEmpty {
      movingVarianceSlices = [ContiguousArray<Tensor.Scalar>](repeating: ContiguousArray(repeating: 1, count: sliceSize), count: inputDim)
    }
  }
  
  private func resetDeltas() {
    let inputDim = inputSize.depth
    dGamma = [Tensor.Scalar](repeating: 0, count: inputDim)
    dBeta = [Tensor.Scalar](repeating: 0, count: inputDim)
  }
  
  private func calculateWelfordVariance(inputs: Tensor, context: NetworkContext) {
    guard isTraining else { return }
    
    updateLock.with {
      welfordVariance.update(inputs)
      
      if welfordVariance.iterations > batchSize {
        fatalError()
      }
    }
  }
  
  private func normalize3DFlat(inputs: Tensor, context: NetworkContext) -> (output: ContiguousArray<Tensor.Scalar>, normalized: [NormalizationFlat]) {
    let depth = inputs.depthSliceCount
    let sliceSize = inputSize.rows * inputSize.columns
    var outStorage = ContiguousArray<Tensor.Scalar>(repeating: 0, count: inputs.storage.count)
    var normalizedInputs: [NormalizationFlat] = []
    
    for i in 0..<depth {
      let inputSlice = inputs.depthSlice(i)
      let outOffset = i * sliceSize
      
      if isTraining {
        updateLock.with {
          let (mean, variance, std, normalized) = normalize2DFlat(inputs: inputSlice, index: i,
                                                                   batchSize: context.totalInBatch)
          normalizedInputs.append(NormalizationFlat(normalized: normalized, std: std))
          
          // gamma[i] * normalized + beta[i]
          let scaled = NumSwiftFlat.add(NumSwiftFlat.multiply(normalized, scalar: gamma[i]), scalar: beta[i])
          
          // Update moving stats
          movingMeanSlices[i] = NumSwiftFlat.add(NumSwiftFlat.multiply(movingMeanSlices[i], scalar: momentum),
                                                  NumSwiftFlat.multiply(mean, scalar: 1 - momentum))
          movingVarianceSlices[i] = NumSwiftFlat.add(NumSwiftFlat.multiply(movingVarianceSlices[i], scalar: momentum),
                                                      NumSwiftFlat.multiply(variance, scalar: 1 - momentum))
          
          for j in 0..<sliceSize { outStorage[outOffset + j] = scaled[j] }
        }
      } else {
        let mm = movingMeanSlices[i]
        let mv = movingVarianceSlices[i]
        // std = sqrt(mv + e)
        let std = NumSwiftFlat.sqrt(NumSwiftFlat.add(mv, scalar: e))
        // normalized = (input - mm) / std
        let normalized = NumSwiftFlat.divide(NumSwiftFlat.subtract(inputSlice, mm), std)
        // output = gamma[i] * normalized + beta[i]
        let scaled = NumSwiftFlat.add(NumSwiftFlat.multiply(normalized, scalar: gamma[i]), scalar: beta[i])
        
        for j in 0..<sliceSize { outStorage[outOffset + j] = scaled[j] }
      }
    }
    
    return (outStorage, normalizedInputs)
  }
  
  private func normalize2DFlat(inputs: ContiguousArray<Tensor.Scalar>,
                                index: Int,
                                batchSize: Int) -> (mean: ContiguousArray<Tensor.Scalar>,
                                                    variance: ContiguousArray<Tensor.Scalar>,
                                                    std: ContiguousArray<Tensor.Scalar>,
                                                    out: ContiguousArray<Tensor.Scalar>) {
    let mean = welfordVariance.means[index]
    let variance = NumSwiftFlat.divide(welfordVariance.m2s[index], scalar: Tensor.Scalar(batchSize))
    let std = NumSwiftFlat.sqrt(NumSwiftFlat.add(variance, scalar: e))
    let normalized = NumSwiftFlat.divide(NumSwiftFlat.subtract(inputs, mean), std)
    
    return (mean, variance, std, normalized)
  }
  
  private func backwardFlat(inputs: Tensor,
                             gradient: Tensor,
                             context: NetworkContext,
                             normalizations: [NormalizationFlat]) -> Tensor {
    let depth = inputs.depthSliceCount
    let sliceSize = inputSize.rows * inputSize.columns
    var outStorage = ContiguousArray<Tensor.Scalar>(repeating: 0, count: inputs.storage.count)
    
    for i in 0..<depth {
      let N = Tensor.Scalar(context.totalInBatch)
      let gradSlice = gradient.depthSlice(i)
      
      var normalized: ContiguousArray<Tensor.Scalar>
      var std: ContiguousArray<Tensor.Scalar>
      
      if let norm = normalizations[safe: i] {
        normalized = norm.normalized
        std = norm.std
      } else {
        let inputSlice = inputs.depthSlice(i)
        let (_, _, nStd, nNormalized) = normalize2DFlat(inputs: inputSlice, index: i, batchSize: context.totalInBatch)
        normalized = nNormalized
        std = nStd
      }
      
      updateLock.with {
        dGamma[i] += NumSwiftFlat.sum(NumSwiftFlat.multiply(gradSlice, normalized))
        dBeta[i] += NumSwiftFlat.sum(gradSlice)
      }
      
      // dxNorm = gradient[i] * gamma[i]
      let dxNorm = NumSwiftFlat.multiply(gradSlice, scalar: gamma[i])
      
      // dx = 1 / N / std * (N * dxNorm - dxNorm.sum - normalized * (dxNorm * normalized).sum)
      let dxNormSum = NumSwiftFlat.sum(dxNorm)
      let dxNormTimesNorm = NumSwiftFlat.multiply(dxNorm, normalized)
      let dxNormTimesNormSum = NumSwiftFlat.sum(dxNormTimesNorm)
      
      let term1 = NumSwiftFlat.multiply(dxNorm, scalar: N)
      let term2 = NumSwiftFlat.subtract(term1, scalar: dxNormSum)
      let term3 = NumSwiftFlat.subtract(term2, NumSwiftFlat.multiply(normalized, scalar: dxNormTimesNormSum))
      let invNStd = NumSwiftFlat.divide(NumSwiftFlat.divide(ContiguousArray(repeating: 1, count: sliceSize), scalar: N), std)
      let dx = NumSwiftFlat.multiply(invNStd, term3)
      
      let outOffset = i * sliceSize
      for j in 0..<sliceSize { outStorage[outOffset + j] = dx[j] }
    }
    
    return Tensor(outStorage, size: inputs.size)
  }
  
}


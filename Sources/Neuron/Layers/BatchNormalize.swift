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
  public var movingMean: [[[Tensor.Scalar]]] = []
  public var movingVariance: [[[Tensor.Scalar]]] = []
  
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

  private var cachedNormalizations: ThreadStorage<UUID, [Normalization]> = .init(defaultValue: [])

  private class Normalization {
    let value: [[Tensor.Scalar]]
    let std: [[Tensor.Scalar]]
    
    init(value: [[Tensor.Scalar]], std: [[Tensor.Scalar]]) {
      self.value = value
      self.std = std
    }
  }
  
  /// Default initializer for Batch Normalize layer
  /// - Parameters:
  ///   - gamma: The gamma property for normalization
  ///   - beta: The beta property for normalization
  ///   - momentum: The momentum property for normalization
  ///   - movingMean: Optional param to set the `movingMean` for the normalizer to start with
  ///   - movingVariance: Optional param to set the `movingVariance` for the normalizer to start with
  ///   - inputSize: Optional param to set the `inputSize` of this layer. [columns, rows, depth]
  public init(gamma: [Tensor.Scalar] = [],
              beta: [Tensor.Scalar] = [],
              momentum: Tensor.Scalar = 0.99,
              movingMean: Tensor.Data = [],
              movingVariance: Tensor.Data = [],
              inputSize: TensorSize? = nil) {
    self.gamma = gamma
    self.beta = beta
    self.movingVariance = movingVariance
    self.movingMean = movingMean
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
    
    try container.encode(movingMean, forKey: .movingMean)
    try container.encode(movingVariance, forKey: .movingVariance)
  }
  
  // actual forward pass happens here from the super class
  public override func performThreadBatchingForwardPass(tensor: Tensor, context: NetworkContext) {
    calculateWelfordVariance(inputs: tensor, context: context)
  }

  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let tensorContext = TensorContext { inputs, gradient in
      let backward = self.backward(inputs: inputs,
                                   gradient: gradient.value,
                                   context: context)
      return (Tensor(backward), Tensor(), Tensor())
    }
    
    let forward = normalize3D(inputs: tensor, context: context)
    let out = Tensor(forward, context: tensorContext)
    
    out.setGraph(tensor)

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
    cachedNormalizations.clear()
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
    
    if gamma.isEmpty {
      self.gamma = [Tensor.Scalar](repeating: 1, count: inputDim)
    }
    
    if beta.isEmpty {
      self.beta = [Tensor.Scalar](repeating: 0, count: inputDim)
    }
  
    if movingMean.isEmpty {
      movingMean = NumSwift.zerosLike((inputSize.rows, inputSize.columns, inputSize.depth))
    }
    
    if movingVariance.isEmpty {
      movingVariance = NumSwift.onesLike((inputSize.rows, inputSize.columns, inputSize.depth))
    }
  }
  
  private func resetDeltas() {
    let inputDim = inputSize.depth
    dGamma = [Tensor.Scalar](repeating: 0, count: inputDim)
    dBeta = [Tensor.Scalar](repeating: 0, count: inputDim)
  }
  
  // TODO: breakout into separate class
  private func calculateWelfordVariance(inputs: Tensor, context: NetworkContext) {
    updateLock.with {
      welfordVariance.update(inputs)
    }
  }
  
  private func normalize3D(inputs:  Tensor, context: NetworkContext) -> [[[Tensor.Scalar]]] {
    var forward: [[[Tensor.Scalar]]] = []

    var normalizedInputs: [Normalization] = []
  
    for i in 0..<inputs.value.count {
      var output: [[Tensor.Scalar]] = []
      if isTraining {
        updateLock.with {
          let (mean, variance, std, normalized) = normalize2D(inputs: inputs.value[i],
                                                              index: i,
                                                              batchSize: context.totalInBatch)
          normalizedInputs.append(.init(value: normalized, std: std))
          
          let normalizedScaledAndShifted = gamma[i] * normalized + beta[i]
                    
          movingMean[i] = momentum * movingMean[i] + (1 - momentum) * mean
          movingVariance[i] = momentum *  movingVariance[i] + (1 - momentum) * variance
          
          output = normalizedScaledAndShifted
        }
      } else {
        let threadMovingMean = movingMean[i]
        let threadMovingVariance = Tensor(movingVariance[i])
        
        let normalized = (Tensor((inputs.value[i] - threadMovingMean)) / threadMovingVariance.sqrt(adding: e)).value[safe: 0] ?? []
        output = gamma[i] * normalized + beta[i]
      }
      
      forward.append(output)
    }
    
    if isTraining {
      cachedNormalizations[inputs.id] = normalizedInputs
    }
    
    return forward
  }
  
  private func normalize2D(inputs: [[Tensor.Scalar]],
                           index: Int,
                           batchSize: Int) -> (mean: [[Tensor.Scalar]],
                                               variance: [[Tensor.Scalar]],
                                               std: [[Tensor.Scalar]],
                                               out:[[Tensor.Scalar]]) {
    
    let mean = welfordVariance.means[index]
      
    let variance = welfordVariance.m2s[index] / batchSize.asTensorScalar
    
    let std: [[Tensor.Scalar]] = Tensor(variance).sqrt(adding: e).value[safe: 0] ?? []
    
    let normalized = (inputs - mean) / std
    
    return  (mean, variance, std, normalized)
  }
  
  private func backward(inputs: Tensor,
                        gradient: [[[Tensor.Scalar]]],
                        context: NetworkContext) -> [[[Tensor.Scalar]]] {
    var backward: [[[Tensor.Scalar]]] = []
    
    let cachedNormalization = cachedNormalizations[inputs.id]
    
    for i in 0..<inputs.value.count {
      let N = Tensor.Scalar(gradient[i].count)
      
      var normalized = cachedNormalization?[safe: i]?.value
      var std = cachedNormalization?[safe: i]?.std
      
      if normalized == nil || std == nil {
        let (_, _, nStd, nNormalized) = normalize2D(inputs: inputs.value[i],
                                                    index: i,
                                                    batchSize: context.totalInBatch)
        
        normalized = nNormalized
        std = nStd
      }
      
      guard let normalized, let std else {
        return []
      }
            
      updateLock.with {
        dGamma[i] += (gradient[i] * normalized).sum
        dBeta[i] += gradient[i].sum
      }

      let dxNorm = gradient[i] * gamma[i]
      
      let dx = 1 / N / std * (N * dxNorm -
                              dxNorm.sum -
                              normalized * (dxNorm * normalized).sum)
      
      backward.append(dx)
    }
  
    return backward
  }
  
}


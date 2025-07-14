//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Performs a normalization of the inputs based on the batch.
public final class BatchNormalize: BaseLayer {
  public var gamma: [Tensor.Scalar] = []
  public var beta: [Tensor.Scalar] = []
//  public var movingMean_DEP: ThreadStorage<UUID,[Tensor.Scalar]>
//  public var movingVariance_DEP: ThreadStorage<UUID, [Tensor.Scalar]>
  public var partialSum: ThreadStorage<UUID, Tensor.Scalar> = .init(defaultValue: 0)
  
  @Atomic public var movingMean: [Tensor.Scalar] = []
  @Atomic public var movingVariance: [Tensor.Scalar] = []

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

  private let e: Tensor.Scalar = 1e-5 //this is a standard smoothing term
  private var dGamma: [Tensor.Scalar] = []
  private var dBeta: [Tensor.Scalar] = []
  @Atomic private var iterations: Int = 0
  @Atomic private var partialIterations: Int = 0
  private let condition = NSCondition()
  
  private var cachedNormalizations: ThreadStorage<UUID, [Normalization]> = .init(defaultValue: [])

  private class Normalization {
    let value: [[Tensor.Scalar]]
    let std: Tensor.Scalar
    
    init(value: [[Tensor.Scalar]], std: Tensor.Scalar) {
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
              movingMean: [Tensor.Scalar] = [],
              movingVariance: [Tensor.Scalar] = [],
              inputSize: TensorSize = TensorSize(array: [])) {
    self.gamma = gamma
    self.beta = beta
    self.movingVariance = movingVariance
    self.movingMean = movingMean
    self.momentum = momentum
    
    super.init(inputSize: inputSize,
               encodingType: .batchNormalize)
    
    setupTrainables()
    resetDeltas()
  }
  
  public enum CodingKeys: String, CodingKey {
    case gamma, beta, momentum, movingMean, movingVariance, inputSize
  }

  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let movingMean = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .movingMean) ?? []
    let movingVar = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .movingVariance) ?? []
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
    let avgDGamma = dGamma / iterations.asTensorScalar
    let avgDBeta = dBeta / iterations.asTensorScalar
    
    gamma = gamma - (avgDGamma * learningRate)
    beta = beta - (avgDBeta * learningRate)
    
    resetDeltas()
    iterations = 0
    cachedNormalizations.clear()
    partialSum.clear()
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
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
  }
  
  private func resetDeltas() {
    let inputDim = inputSize.depth
    dGamma = [Tensor.Scalar](repeating: 0, count: inputDim)
    dBeta = [Tensor.Scalar](repeating: 0, count: inputDim)
  }
  
  private func normalize3D(inputs:  Tensor, context: NetworkContext) -> [[[Tensor.Scalar]]] {
    var forward: [[[Tensor.Scalar]]] = []

    var normalizedInputs: [Normalization] = []
    
    var runningMean: [Tensor.Scalar] = []
    var runningVariance: [Tensor.Scalar] =  []
    
    for i in 0..<inputs.value.count {
      var output: [[Tensor.Scalar]] = []
      if isTraining {
        
        if iterations < batchSize {
          condition.lock()
          
          while partialIterations < context.batchProcessingCount {
            let sum = inputs.value[i].sum
            
            if let currentSum = partialSum[context.threadId] {
              partialSum[context.threadId] = currentSum + sum
            } else {
              partialSum[context.threadId] = sum
            }
            
            partialIterations += 1
            iterations += 1

            condition.broadcast()
          }
          
          partialIterations = 0
          
          // not just batchSize because the batch could be smaller than 32
          while iterations < context.totalInBatch {
            condition.wait()
          }
          
          condition.unlock()
        }
                
        // todo: support multiple batches in 1 tensor
        let (mean, variance, std, normalized) = normalize2D(inputs: inputs.value[i], batchSize: context.totalInBatch)
        normalizedInputs.append(.init(value: normalized, std: std))
        
        let normalizedScaledAndShifted = gamma[i] * normalized + beta[i]
        
        let threadMovingMean = movingMean[safe: i] ?? 0.0
        let threadMovingVariance = movingVariance[safe: i] ?? 1.0
        
        runningMean.append(momentum * threadMovingMean + (1 - momentum) * mean)
        runningVariance.append(momentum * threadMovingVariance + (1 - momentum) * variance)
    
        output = normalizedScaledAndShifted
      } else {
        let threadMovingMean = movingMean[safe: i] ?? 0.0
        let threadMovingVariance = movingVariance[safe: i] ?? 1.0
        
        let normalized = (inputs.value[i] - threadMovingMean) / sqrt(threadMovingVariance + e)
        output = gamma[i] * normalized + beta[i]
      }
      
      forward.append(output)
    }
    
    if isTraining {
      movingMean = runningMean
      movingVariance = runningVariance
      
      cachedNormalizations[inputs.id] = normalizedInputs
    }
    
    return forward
  }
  
  private func normalize2D(inputs: [[Tensor.Scalar]], batchSize: Int) -> (mean: Tensor.Scalar, variance: Tensor.Scalar, std: Tensor.Scalar, out:[[Tensor.Scalar]]) {
    
    let mean = partialSum.reduceMean(batchSize: batchSize.asTensorScalar)
      
    let variance = partialSum.reduceVariance(mean: mean, batchSize: batchSize.asTensorScalar)
    
    let std = sqrt(variance + e)
    
    let normalized = (inputs - mean) / std
    
    return  (mean, variance, std, normalized)
  }
  
  private func backward(inputs: Tensor,
                        gradient: [[[Tensor.Scalar]]],
                        context: NetworkContext) -> [[[Tensor.Scalar]]] {
    var backward: [[[Tensor.Scalar]]] = []
    
    // it's necessarily going to guarentee that this is run on the same thread that called forward..
    let cachedNormalization = cachedNormalizations[inputs.id]
    
    for i in 0..<inputs.value.count {
      let N = Tensor.Scalar(gradient[i].count)
      
      var normalized = cachedNormalization?[safe: i]?.value
      var std = cachedNormalization?[safe: i]?.std
      
      if normalized == nil || std == nil {
        let (_, _, nStd, nNormalized) = normalize2D(inputs: inputs.value[i], batchSize: context.totalInBatch)
        
        normalized = nNormalized
        std = nStd
      }
      
      guard let normalized, let std else {
        return []
      }
      
      let lock = NSLock()
      
      lock.with {
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


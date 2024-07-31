//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Performs a normalization of the inputs based on the batch.
public final class BatchNormalize<N: TensorNumeric>: BaseLayer<N> {
  public var gamma: [Tensor<N>.Scalar] = []
  public var beta: [Tensor<N>.Scalar] = []
  public var movingMean: [Tensor<N>.Scalar] = []
  public var movingVariance: [Tensor<N>.Scalar] = []
  public let momentum: Tensor<N>.Scalar
  public override var weights: Tensor<N> {
    // only used for print purposes.
    get {
      var beta = beta
      beta.append(contentsOf: gamma)
      beta.append(contentsOf: movingMean)
      beta.append(contentsOf: movingVariance)
      return Tensor<N>(beta)
    }
    set {}
  }

  private let e: Tensor<N>.Scalar = 1e-5 //this is a standard smoothing term
  private var dGamma: [Tensor<N>.Scalar] = []
  private var dBeta: [Tensor<N>.Scalar] = []
  @Atomic private var iterations: Int = 0
  
  /// Default initializer for Batch Normalize layer
  /// - Parameters:
  ///   - gamma: The gamma property for normalization
  ///   - beta: The beta property for normalization
  ///   - momentum: The momentum property for normalization
  ///   - movingMean: Optional param to set the `movingMean` for the normalizer to start with
  ///   - movingVariance: Optional param to set the `movingVariance` for the normalizer to start with
  ///   - inputSize: Optional param to set the `inputSize` of this layer. [columns, rows, depth]
  public init(gamma: [Tensor<N>.Scalar] = [],
              beta: [Tensor<N>.Scalar] = [],
              momentum: Tensor<N>.Scalar = 0.99,
              movingMean: [Tensor<N>.Scalar] = [],
              movingVariance: [Tensor<N>.Scalar] = [],
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
    let movingMean = try container.decodeIfPresent([Tensor<N>.Scalar].self, forKey: .movingMean) ?? []
    let movingVar = try container.decodeIfPresent([Tensor<N>.Scalar].self, forKey: .movingVariance) ?? []
    let gamma = try container.decodeIfPresent([Tensor<N>.Scalar].self, forKey: .gamma) ?? []
    let beta = try container.decodeIfPresent([Tensor<N>.Scalar].self, forKey: .beta) ?? []
    let momentum = try container.decodeIfPresent(Tensor<N>.Scalar.self, forKey: .momentum) ?? 0.99

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
    try container.encode(movingMean, forKey: .movingMean)
    try container.encode(movingVariance, forKey: .movingVariance)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(momentum, forKey: .momentum)
  }

  public override func forward(tensor: Tensor<N>) -> Tensor<N> {
    let context = TensorContext<N> { inputs, gradient in
      let backward = self.backward(inputs: inputs.value, gradient: gradient.value)
      return (Tensor<N>(backward), Tensor<N>(), Tensor<N>())
    }
    
    let forward = normalize(inputs: tensor.value)
    let out = Tensor<N>(forward, context: context)
    
    out.setGraph(tensor)

    return out
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor<N>.Scalar) {
    gamma = gamma - (dGamma / iterations.asTensorScalar)
    beta = beta - (dBeta / iterations.asTensorScalar)
    
    resetDeltas()
    iterations = 0
  }
  
  override public func onInputSizeSet() {
    outputSize = inputSize
    setupTrainables()
    resetDeltas()
  }
  
  private func setupTrainables() {
    let inputDim = inputSize.depth
    
    if gamma.isEmpty {
      self.gamma = [Tensor<N>.Scalar](repeating: 1, count: inputDim)
    }
    
    if beta.isEmpty {
      self.beta = [Tensor<N>.Scalar](repeating: 0, count: inputDim)
    }
    
    if movingMean.isEmpty {
      self.movingMean = [Tensor<N>.Scalar](repeating: 0, count: inputDim)
    }
    
    if movingVariance.isEmpty {
      self.movingVariance = [Tensor<N>.Scalar](repeating: 1, count: inputDim)
    }
  }
  
  private func resetDeltas() {
    let inputDim = inputSize.depth
    dGamma = [Tensor<N>.Scalar](repeating: 0, count: inputDim)
    dBeta = [Tensor<N>.Scalar](repeating: 0, count: inputDim)
  }
  
  private func normalize(inputs: [[[Tensor<N>.Scalar]]]) -> [[[Tensor<N>.Scalar]]] {
    var forward: [[[Tensor<N>.Scalar]]] = []

    for i in 0..<inputs.count {
      let count = inputSize.rows * inputSize.columns
      let total = Tensor<N>.Scalar(count)
      
      let mean = isTraining == true ? inputs[i].sum / total : movingMean[i]
      
      let inputsCentered = inputs[i] - mean

      let variance = isTraining == true ? inputsCentered.sumOfSquares / total : movingVariance[i]
              
      let std = sqrt(variance + e)
      
      let normalized = inputsCentered / std
      
      let normalizedScaledAndShifted = normalized * gamma[i] + beta[i]

      if isTraining {
        let lock = NSLock()
        lock.with {
          movingMean[i] = momentum * movingMean[i] + (1 - momentum) * mean
          movingVariance[i] = momentum * movingVariance[i] + (1 - momentum) * variance
        }
      }
      
      forward.append(normalizedScaledAndShifted)
    }
    
    if isTraining {
      iterations += 1
    }

    return forward
  }
  
  private func backward(inputs: [[[Tensor<N>.Scalar]]], gradient: [[[Tensor<N>.Scalar]]]) -> [[[Tensor<N>.Scalar]]] {
    // we're doing normalization again to support multithreading.
    // TODO: figure out a way to not have to do this math again
    
    var backward: [[[Tensor<N>.Scalar]]] = []
    
    for i in 0..<inputs.count {
      let count = inputSize.rows * inputSize.columns
      let total = Tensor<N>.Scalar(count)

      let N = Tensor<N>.Scalar(gradient[i].count)
      
      let mean = inputs[i].sum / total
      
      let inputsCentered = inputs[i] - mean

      let variance = inputsCentered.sumOfSquares / total
              
      let std = sqrt(variance + e)
      
      let normalized = inputsCentered / std
      
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


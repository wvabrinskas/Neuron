//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Performs a normalization of the inputs based on the batch.
public final class BatchNormalize: Layer {
  public var encodingType: EncodingType = .batchNormalize
  public var device: Device = CPU()
  public var biasEnabled: Bool = true
  public var trainable: Bool = true
  public var inputSize: TensorSize = TensorSize(array: []) {
    didSet {
      outputSize = inputSize
      setupTrainables()
      resetDeltas()
    }
  }
  public var outputSize: TensorSize = TensorSize(array: [])
  public var weights: Tensor {
    // For printing purposes. Not actually used
    var trainables = beta
    trainables.append(contentsOf: gamma)
    return Tensor(trainables)
  }
  public var biases: Tensor = Tensor()
  public var initializer: Initializer?
  
  public var gamma: [Tensor.Scalar] = []
  public var beta: [Tensor.Scalar] = []
  public var movingMean: [Tensor.Scalar] = []
  public var movingVariance: [Tensor.Scalar] = []
  public let momentum: Tensor.Scalar

  private let e: Float = 1e-5 //this is a standard smoothing term
  private var dGamma: [Tensor.Scalar] = []
  private var dBeta: [Tensor.Scalar] = []
  @AtomicCodable private var iterations: Int = 0
  
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
    self.inputSize = inputSize

    setupTrainables()
    resetDeltas()
  }
  
  public enum CodingKeys: String, CodingKey {
    case gamma, beta, momentum, movingMean, movingVariance, inputSize
  }

  convenience public init(from decoder: Decoder) throws {
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
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(movingMean, forKey: .movingMean)
    try container.encode(movingVariance, forKey: .movingVariance)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(momentum, forKey: .momentum)
  }

  public func forward(tensor: Tensor) -> Tensor {
    let context = TensorContext { inputs, gradient in
      let backward = self.backward(inputs: inputs.value, gradient: gradient.value)
      return (Tensor(backward), Tensor())
    }
    
    let forward = normalize(inputs: tensor.value)
    return Tensor(forward, context: context)
  }
  
  public func apply(gradients: Optimizer.Gradient) {
    gamma = gamma - (dGamma / iterations.asTensorScalar)
    beta = beta - (dBeta / iterations.asTensorScalar)
    
    resetDeltas()
    iterations = 0
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
      self.movingMean = [Tensor.Scalar](repeating: 0, count: inputDim)
    }
    
    if movingVariance.isEmpty {
      self.movingVariance = [Tensor.Scalar](repeating: 1, count: inputDim)
    }
  }
  
  private func resetDeltas() {
    let inputDim = inputSize.depth
    dGamma = [Tensor.Scalar](repeating: 0, count: inputDim)
    dBeta = [Tensor.Scalar](repeating: 0, count: inputDim)
  }
  
  private func normalize(inputs: [[[Tensor.Scalar]]]) -> [[[Tensor.Scalar]]] {
    var forward: [[[Tensor.Scalar]]] = []

    for i in 0..<inputs.count {
      let count = inputSize.rows * inputSize.columns
      let total = Float(count)
      
      let mean = trainable == true ? inputs[i].sum / total : movingMean[i]
      
      let inputsCentered = inputs[i] - mean

      let variance = trainable == true ? inputsCentered.sumOfSquares / total : movingVariance[i]
              
      let std = sqrt(variance + e)
      
      let normalized = inputsCentered / std
      
      let normalizedScaledAndShifted = normalized * gamma[i] + beta[i]

      if trainable {
        let lock = NSLock()
        lock.with {
          movingMean[i] = momentum * movingMean[i] + (1 - momentum) * mean
          movingVariance[i] = momentum * movingVariance[i] + (1 - momentum) * variance
        }
      }
      
      forward.append(normalizedScaledAndShifted)
    }
    
    if trainable {
      iterations += 1
    }

    return forward
  }
  
  private func backward(inputs: [[[Tensor.Scalar]]], gradient: [[[Tensor.Scalar]]]) -> [[[Tensor.Scalar]]] {
    // we're doing normalization again to support multithreading.
    // TODO: figure out a way to not have to do this math again
    
    var backward: [[[Tensor.Scalar]]] = []
    
    for i in 0..<inputs.count {
      let count = inputSize.rows * inputSize.columns
      let total = Float(count)

      let N = Float(gradient[i].count)
      
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


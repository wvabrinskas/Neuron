//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Batch Normalization layer for stabilizing and accelerating training
/// Normalizes inputs by adjusting and scaling activations based on batch statistics
/// Reduces internal covariate shift and allows higher learning rates
/// Formula: y = γ * (x - μ) / √(σ² + ε) + β
public final class BatchNormalize: BaseLayer {
  /// Scale parameter (γ) learned during training
  public var gamma: [Tensor.Scalar] = []
  /// Shift parameter (β) learned during training
  public var beta: [Tensor.Scalar] = []
  /// Running average of means for inference
  public var movingMean: [Tensor.Scalar] = []
  /// Running average of variances for inference
  public var movingVariance: [Tensor.Scalar] = []
  /// Momentum factor for updating running statistics
  public let momentum: Tensor.Scalar
  /// Combined weights tensor for display purposes
  public override var weights: Tensor {
    // only used for print purposes.
    get {
      var beta = beta
      beta.append(contentsOf: gamma)
      beta.append(contentsOf: movingMean)
      beta.append(contentsOf: movingVariance)
      return Tensor(beta)
    }
    set {}
  }

  /// Small constant for numerical stability
  private let e: Tensor.Scalar = 1e-5
  /// Gradients for gamma parameters
  private var dGamma: [Tensor.Scalar] = []
  /// Gradients for beta parameters
  private var dBeta: [Tensor.Scalar] = []
  /// Number of training iterations for gradient averaging
  @Atomic private var iterations: Int = 0
  
  /// Initializes a Batch Normalization layer
  /// - Parameters:
  ///   - gamma: Scale parameters (γ). Empty array initializes to ones
  ///   - beta: Shift parameters (β). Empty array initializes to zeros
  ///   - momentum: Momentum for updating running statistics. Default: 0.99
  ///   - movingMean: Initial running mean values. Empty array initializes to zeros
  ///   - movingVariance: Initial running variance values. Empty array initializes to ones
  ///   - inputSize: Input tensor dimensions [columns, rows, depth]
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
  
  /// Coding keys for serialization
  public enum CodingKeys: String, CodingKey {
    case gamma, beta, momentum, movingMean, movingVariance, inputSize
  }

  /// Initializes BatchNormalize layer from decoder for deserialization
  /// - Parameter decoder: Decoder containing serialized layer data
  /// - Throws: Decoding errors if deserialization fails
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
    try container.encode(movingMean, forKey: .movingMean)
    try container.encode(movingVariance, forKey: .movingVariance)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(momentum, forKey: .momentum)
  }

  /// Performs forward pass through batch normalization
  /// Normalizes input using batch statistics during training, running stats during inference
  /// - Parameters:
  ///   - tensor: Input tensor to normalize
  ///   - context: Network context for computation
  /// - Returns: Normalized tensor with learnable scale and shift applied
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient in
      let backward = self.backward(inputs: inputs.value, gradient: gradient.value)
      return (Tensor(backward), Tensor(), Tensor())
    }
    
    let forward = normalize(inputs: tensor.value)
    let out = Tensor(forward, context: context)
    
    out.setGraph(tensor)

    return out
  }
  
  /// Applies accumulated gradients to update gamma and beta parameters
  /// Averages gradients over the number of training iterations
  /// - Parameters:
  ///   - gradients: Accumulated gradients (unused for batch norm)
  ///   - learningRate: Learning rate for parameter updates (unused for batch norm)
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    gamma = gamma - (dGamma / iterations.asTensorScalar)
    beta = beta - (dBeta / iterations.asTensorScalar)
    
    resetDeltas()
    iterations = 0
  }
  
  /// Called when input size is set, initializes parameters and gradients
  /// Sets up gamma, beta, moving mean, and moving variance arrays
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
      let total = Tensor.Scalar(count)
      
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
  
  private func backward(inputs: [[[Tensor.Scalar]]], gradient: [[[Tensor.Scalar]]]) -> [[[Tensor.Scalar]]] {
    // we're doing normalization again to support multithreading.
    // TODO: figure out a way to not have to do this math again
    
    var backward: [[[Tensor.Scalar]]] = []
    
    for i in 0..<inputs.count {
      let count = inputSize.rows * inputSize.columns
      let total = Tensor.Scalar(count)

      let N = Tensor.Scalar(gradient[i].count)
      
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


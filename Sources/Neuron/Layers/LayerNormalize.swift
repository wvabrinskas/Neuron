//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/22/22.
//

import Foundation
import NumSwift

/// Performs a layer normalization function.
public final class LayerNormalize: Layer {
  public var encodingType: EncodingType = .layerNormalize
  public var inputSize: TensorSize {
    didSet {
      outputSize = inputSize
    }
  }
  public var outputSize: TensorSize
  public var weights: Tensor {
    // For printing purposes. Not actually used
    var trainables = beta
    trainables.append(contentsOf: gamma)
    return Tensor(trainables)
  }
  public var biases: Tensor = Tensor()
  public var biasEnabled: Bool = false
  public var trainable: Bool = true
  public var initializer: Initializer? = nil
  public var device: Device = CPU()
  public var isTraining: Bool = true

  private var epsilon: Tensor.Scalar
  public var gamma: [Tensor.Scalar] = []
  public var beta: [Tensor.Scalar] = []
  private var dGamma: [Tensor.Scalar] = []
  private var dBeta: [Tensor.Scalar] = []
  @Atomic private var iterations: Int = 0

  public enum CodingKeys: String, CodingKey {
    case gamma, beta, epsilon, inputSize
  }
  
  /// Default initializer for layer normalization.
  /// - Parameters:
  ///   - epsilon: Epsilon value for normalization. Defualt: `1e-10`
  ///   - gamma: Gamme value for normalization
  ///   - beta: Beta value for normalization
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(epsilon: Tensor.Scalar = 1e-10,
              gamma: [Tensor.Scalar] = [],
              beta: [Tensor.Scalar] = [],
              inputSize: TensorSize = TensorSize(array: [])) {
    self.epsilon = epsilon
    self.beta = beta
    self.gamma = gamma
    self.inputSize = inputSize
    self.outputSize = inputSize
    
    setupTrainables()
    resetDeltas()
  }
  
  convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let gamma = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .gamma) ?? []
    let beta = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .beta) ?? []
    let epsilon = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .epsilon) ?? 1e-10

    self.init(epsilon: epsilon,
              gamma: gamma,
              beta: beta)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
    
    setupTrainables()
    resetDeltas()
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(epsilon, forKey: .epsilon)
  }
  
  public func forward(tensor: Tensor) -> Tensor {
    let context = TensorContext { inputs, gradient in
      let gradient = self.backward(inputs: inputs.value, gradient: gradient.value)
      return (Tensor(gradient), Tensor(), Tensor())
    }
    
    let forward = normalize(inputs: tensor.value)
    let out = Tensor(forward, context: context)
    out.setGraph(tensor)
    return out
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
      
      let mean = inputs[i].mean
      let inputsCentered = inputs[i] - mean
      let variance = inputsCentered.sumOfSquares / total
      
      let std = sqrt(variance + epsilon)
      
      var result = (inputs[i] - mean) / std
      result = result * gamma[i] + beta[i]
      forward.append(result)
    }
    
    if trainable {
      iterations += 1
    }

    return forward
  }
  
  private func backward(inputs: [[[Tensor.Scalar]]], gradient: [[[Tensor.Scalar]]]) -> [[[Tensor.Scalar]]] {
    
    var backward: [[[Tensor.Scalar]]] = []
    
    for i in 0..<inputs.count {
      let count = inputSize.rows * inputSize.columns
      let total = Float(count)
      
      let N = Float(gradient[i].count)
      
      let mean = inputs[i].mean
      let inputsCentered = inputs[i] - mean
      let variance = inputsCentered.sumOfSquares / total
      
      let std = sqrt(variance + epsilon)
      
      let scaledX = (inputs[i] - mean)
      let normalized = scaledX / std
      
      let ivar = 1 / std
      let gradientSum = gradient[i].sum
      
      let part1 = (1 / N) * gamma[i]
      let part2 = (1/std * ((N * gradient[i])))
      
      let result = part1 * part2 -
                   gradientSum - ((scaledX) * pow(ivar, 2) *
                   (gradient[i] * scaledX).sum)
      
      backward.append(result)
      
      let lock = NSLock()
      
      lock.with {
        dGamma[i] += (gradient[i] * normalized).sum
        dBeta[i] += gradient[i].sum
      }

    }
    
    return backward
  }
  
  public func apply(gradients: Optimizer.Gradient, learningRate: Float) {
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
  }
}

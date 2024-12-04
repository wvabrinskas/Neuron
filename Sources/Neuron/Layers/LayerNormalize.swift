//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/22/22.
//

import Foundation
import NumSwift

/// Performs a layer normalization function.
public final class LayerNormalize: BaseLayer {
  public override var weights: Tensor {
    get {
      // For printing purposes. Not actually used
      beta.concat(gamma, axis: 2)
    }
    set {}
  }

  private var epsilon: Tensor.Scalar
  public var gamma: Tensor = .init()
  public var beta: Tensor = .init()
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
  public init(epsilon: Tensor.Scalar = .stabilityFactor,
              gamma: Tensor = .init(),
              beta: Tensor = .init(),
              inputSize: TensorSize = TensorSize(array: [])) {
    self.epsilon = epsilon
    self.beta = beta
    self.gamma = gamma
    
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: .layerNormalize)
    
    self.outputSize = inputSize
    
    setupTrainables()
  }
  
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let gamma = try container.decodeIfPresent(Tensor.self, forKey: .gamma) ?? .init()
    let beta = try container.decodeIfPresent(Tensor.self, forKey: .beta) ?? .init()
    let epsilon = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .epsilon) ?? .stabilityFactor

    self.init(epsilon: epsilon,
              gamma: gamma,
              beta: beta)
    
    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize
    
    setupTrainables()
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(epsilon, forKey: .epsilon)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient in
      self.backward(inputs: inputs.value, gradient: gradient.value)
    }
    
    let forward = normalize(inputs: tensor.value)
    let out = Tensor(forward, context: context)
    out.setGraph(tensor)
    return out
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    setupTrainables()
  }

  private func normalize(inputs: [[[Tensor.Scalar]]]) -> [[[Tensor.Scalar]]] {
    
    var forward: [[[Tensor.Scalar]]] = []
    
    for i in 0..<inputs.count {
      let count = inputSize.rows * inputSize.columns
      let total = Tensor.Scalar(count)
      
      let mean = inputs[i].mean
      let inputsCentered = inputs[i] - mean
      let variance = inputsCentered.sumOfSquares / total
      
      let std = sqrt(variance + epsilon)
      
      var result = (inputs[i] - mean) / std
      result = result * gamma.value[i] + beta.value[i]
      forward.append(result)
    }

    return forward
  }
  
  private func backward(inputs: [[[Tensor.Scalar]]], gradient: [[[Tensor.Scalar]]]) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    var dInputs: Tensor.Data = []
    var dGamma: Tensor.Data = []
    var dBeta: Tensor.Data = []
    
    for i in 0..<inputs.count {
      let feature = inputs[i]
      let gradient = gradient[i]
      
      let count = inputSize.rows
      let N = Tensor.Scalar(count)
      
      let mean = Tensor(feature).mean(axis: 1).value
      let variance = Tensor(feature).variance(axis: 1)
      let varianceEpsilon = variance + epsilon
      let std = (varianceEpsilon).sqrt()
      
      let inputsMinusMean = Tensor(feature) - Tensor(mean)

      let x_norm = inputsMinusMean / std

      let dL_dbeta = Tensor(gradient).sum(axis: 2)
      
      let dL_dgamma = (x_norm * Tensor(gradient)).sum(axis: 2)
      
      let line1 = Tensor(gamma.value[i]) * Tensor((1 / (N * std.value)))
      let line2 = Tensor(N * gradient)
      let line3 = dL_dbeta
      
      let line4 = inputsMinusMean / varianceEpsilon
      let line5 = (Tensor(feature) - Tensor(gradient) * Tensor(mean)).sum(axis: 2)
      
      let dl_dx = line1 * (
        line2 - line3
        - line4 * line5
      )
      
      dInputs.append(dl_dx.value[safe: 0, []])
      dGamma.append(dL_dgamma.value[safe: 0, []])
      dBeta.append(dL_dbeta.value[safe: 0, []])
    }
    
    return (Tensor(dInputs), Tensor(dGamma).concat(Tensor(dBeta), axis: 2), Tensor())
  }
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    let gammaWeights = gradients.weights[0..., 0..., 0..<inputSize.depth]
    let betaWeights = gradients.weights[0..., 0..., inputSize.depth...]

    gamma = gamma - gammaWeights
    beta = beta - betaWeights
  }
  
  private func setupTrainables() {
    if gamma.isEmpty {
      self.gamma = Tensor(NumSwift.onesLike((rows: inputSize.rows, columns: inputSize.columns, depth: inputSize.depth)))
    }
    
    if beta.isEmpty {
      self.beta = Tensor(NumSwift.zerosLike((rows: inputSize.rows, columns: inputSize.columns, depth: inputSize.depth)))
    }
  }
}

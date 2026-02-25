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
  /// The combined weights tensor of beta and gamma, used primarily for display and inspection purposes.
  /// - Returns: A `Tensor` containing the concatenated beta and gamma values shaped to the input size.
  public override var weights: Tensor {
    get {
      // For printing purposes. Not actually used
      let out = beta.concat(gamma, axis: 2)
      
      let outTensor = Tensor(out.storage, size: .init(rows: beta.size.rows,
                                                      columns: beta.size.columns,
                                                      depth: beta.size.depth + gamma.size.depth))
      return outTensor
    }
    set {}
  }

  private var epsilon: Tensor.Scalar
  /// The gamma scaling parameter used in layer normalization.
  public var gamma: Tensor = .init()
  /// The beta shift parameter used in layer normalization.
  public var beta: Tensor = .init()

  /// Coding keys used for encoding and decoding the layer normalization layer.
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
              inputSize: TensorSize? = nil) {
    self.epsilon = epsilon
    self.beta = beta
    self.gamma = gamma
    
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: .layerNormalize)
    
    if let inputSize {
      self.outputSize = inputSize
    }
    
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
  
  /// Encodes layer-normalization parameters.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(epsilon, forKey: .epsilon)
  }
  
  /// Applies layer normalization over each depth slice.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Normalized tensor with layer-norm backpropagation context.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let context = TensorContext { inputs, gradient, wrt in
      self.backwardFlat(inputs: inputs, gradient: gradient)
    }
    
    let forwardStorage = normalizeFlat(inputs: tensor)
    let out = Tensor(forwardStorage, size: tensor.size, context: context)
    out.setGraph(tensor)
    return out
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    setupTrainables()
  }

  private func normalizeFlat(inputs: Tensor) -> Tensor.Value {
    let sliceSize = inputSize.rows * inputSize.columns * inputSize.depth
    let total = Tensor.Scalar(sliceSize)
    
    let gammaSlice = gamma
    let betaSlice = beta
    
    let slice = inputs
    let mean = slice.mean().asScalar()
    
    let centered = slice - mean
    let variance = centered.sumOfSquares().asScalar() / total
    let std = Tensor.Scalar.sqrt(variance + epsilon)
    
    let normalized = centered / std
    
    let scaled = normalized * gammaSlice + betaSlice
    
    return scaled.storage
  }
  
  private func backwardFlat(inputs: Tensor, gradient: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let gammaTensor = gamma.asScalar()
    let featureTensor = inputs
    let gradTensor = gradient
    
    let N = Tensor.Scalar(inputSize.rows * inputSize.columns * inputSize.depth)
    
    let mean = featureTensor.mean(axis: -1).asScalar()
    let variance = featureTensor.variance(axis: -1).asScalar()
    let varianceEpsilon = variance + epsilon
    let std = Tensor.Scalar.sqrt(varianceEpsilon)
    
    let inputsMinusMean = featureTensor - mean
    let x_norm = inputsMinusMean / std
    
    // For weight updates - keep feature shape (4, 2, 2)
    let dL_dbeta = gradTensor   // already the right shape if no batch dim
    let dL_dgamma = x_norm * gradTensor

    // Scalars just for the dl_dx formula
    let sumGrad = gradTensor.sum()           // scalar
    let sumGradXnorm = (x_norm * gradTensor).sum()  // scalar
    
    let invNStd = gammaTensor * (Tensor.Scalar(1) / (N * std))
    let line2 = N * gradTensor
    let line3 = sumGrad                          // sum(dOut)
    let line4 = x_norm                            // x̂
    let line5 = sumGradXnorm                         // sum(dOut * x̂)

    let dl_dx = invNStd * (line2 - line3 - line4 * line5)
    
    let deltaWeights = dL_dgamma.concat(dL_dbeta, axis: 2)
        
    return (Tensor(dl_dx.storage, size: inputs.size),
            deltaWeights,
            Tensor())
  }
  
  /// Applies layer-normalization parameter updates.
  ///
  /// - Parameters:
  ///   - gradients: Combined gamma/beta gradients packed in `weights`.
  ///   - learningRate: Learning rate already reflected by optimizer gradients.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    let gammaWeights = gradients.weights.depthSliceTensor(0)
    let betaWeights = gradients.weights.depthSliceTensor(1)

    gamma = gamma - gammaWeights
    beta = beta - betaWeights
  }
  
  private func setupTrainables() {
    if gamma.isEmpty {
      self.gamma = Tensor.fillWith(value: 1.0, size: inputSize)
    }
    
    if beta.isEmpty {
      self.beta = Tensor.fillWith(value: 0.0, size: inputSize)
    }
  }
}

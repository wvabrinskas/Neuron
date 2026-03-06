//
//  InstanceNormalization.swift
//
//

import Foundation
import NumSwift

/// A layer that performs instance normalization on its input.
/// 
/// Instance normalization normalizes each sample in a batch independently
/// across spatial dimensions, using learnable scale (gamma) and shift (beta) parameters.
public final class InstanceNormalize: BaseLayer {
  /// The combined weights tensor of beta and gamma, used primarily for display and inspection purposes.
  /// - Returns: A `Tensor` containing the concatenated beta and gamma values shaped to the input size.
  public override var weights: Tensor {
    get {
      // Must match gradient layout (beta | gamma) for correct optimizer weight decay indexing
      return beta.concat(gamma, axis: 2)
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
    case gamma, beta, epsilon, inputSize, linkId
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
              inputSize: TensorSize? = nil,
              linkId: String = UUID().uuidString) {
    self.epsilon = epsilon
    self.beta = beta
    self.gamma = gamma
    
    super.init(inputSize: inputSize,
               biasEnabled: false,
               linkId: linkId,
               encodingType: .instanceNorm)
    
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

    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    
    self.init(epsilon: epsilon,
              gamma: gamma,
              beta: beta,
              linkId: linkId)
    
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
    try container.encode(linkId, forKey: .linkId)
  }
  
  /// Applies layer normalization over each depth slice.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Normalized tensor with layer-norm backpropagation context.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let tensorContext = TensorContext { inputs, gradient, wrt in
      self.backwardFlat(inputs: inputs, gradient: gradient)
    }
    
    let forwardStorage = normalizeFlat(inputs: tensor)
    let out = Tensor(forwardStorage, size: tensor.size, context: tensorContext)
    out.setGraph(tensor)
    
    return super.forward(tensor: out, context: context)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    setupTrainables()
  }

  private func normalizeFlat(inputs: Tensor) -> Tensor.Value {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let total = Tensor.Scalar(sliceSize)
    var outStorage = Tensor.Value(repeating: 0, count: inputs.storage.count)
    
    for i in 0..<depth {
      let gammaSlice = gamma.storage[i]
      let betaSlice = beta.storage[i]
      
      let slice = inputs.depthSlice(i)
      let mean = slice.mean
      
      let centered = slice - mean
      let variance = centered.sumOfSquares / total
      let std = Tensor.Scalar.sqrt(variance + epsilon)
      
      let normalized = centered / std
      // result = normalized * gamma[i] + beta[i]
      let scaled = normalized * gammaSlice + betaSlice
      
      let offset = i * sliceSize
      for j in 0..<sliceSize { outStorage[offset + j] = scaled[j] }
    }
    
    return outStorage
  }
  
  private func backwardFlat(inputs: Tensor, gradient: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let depth = inputs.size.depth
    
    // We use Tensor operations per-depth for the complex backward math
    // but construct depth-1 Tensors from flat slices instead of going through .value
    let sliceSize = inputSize.rows * inputSize.columns
    let dInputResult = TensorStorage.create(count: sliceSize * depth)
    let dGammaResult = TensorStorage.create(count: depth)
    let dBetaResult = TensorStorage.create(count: depth)
    
    for i in 0..<depth {
      let gammaTensor = gamma.storage[i]
      let featureTensor = inputs.depthSliceTensor(i)
      let gradTensor = gradient.depthSliceTensor(i)
      
      let N = Tensor.Scalar(inputSize.rows * inputSize.columns)
      
      let mean = featureTensor.mean(axis: -1).asScalar()
      let variance = featureTensor.variance(axis: -1).asScalar()
      let varianceEpsilon = variance + epsilon
      let std = Tensor.Scalar.sqrt(varianceEpsilon)
      
      let inputsMinusMean = featureTensor - mean
      let x_norm = inputsMinusMean / std
      
      let dL_dbeta = gradTensor.sum(axis: -1).asScalar()
      let dL_dgamma = (x_norm * gradTensor).sum(axis: -1).asScalar()
      
      let invNStd = gammaTensor * (Tensor.Scalar(1) / (N * std))
      let line2 = N * gradTensor
      let line3 = dL_dbeta
      let line4 = x_norm
      let line5 = dL_dgamma

      let dl_dx = invNStd * (line2 - line3 - line4 * line5)
      
      let offset = i * sliceSize
      for j in 0..<dl_dx.storage.count {
        dInputResult[offset + j] = dl_dx.storage[j]
      }
      dGammaResult[i] = dL_dgamma
      dBetaResult[i] = dL_dbeta
    }
    
    let dGammaTensor = Tensor(storage: dGammaResult, size: TensorSize(rows: 1, columns: depth, depth: 1))
    let dBetaTensor = Tensor(storage: dBetaResult, size: TensorSize(rows: 1, columns: depth, depth: 1))
    
    // Return dBeta.concat(dGamma) to match weights layout (beta.concat(gamma)) for correct optimizer indexing
    return (Tensor(storage: dInputResult, size: inputs.size),
            dBetaTensor.concat(dGammaTensor, axis: 2),
            Tensor())
  }
  
  /// Applies layer-normalization parameter updates.
  ///
  /// - Parameters:
  ///   - gradients: Combined gamma/beta gradients packed in `weights`.
  ///   - learningRate: Learning rate already reflected by optimizer gradients.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    // Gradients match weights layout: beta | gamma (depth 0 = beta, depth 1 = gamma)
    let betaWeights = gradients.weights.depthSliceTensor(0)
    let gammaWeights = gradients.weights.depthSliceTensor(1)

    gamma = gamma - gammaWeights
    beta = beta - betaWeights
  }
  
  private func setupTrainables() {
    if gamma.isEmpty {
      self.gamma = Tensor.fillWith(value: 1.0, size: .init(rows: 1, columns: inputSize.depth, depth: 1))
    }
    
    if beta.isEmpty {
      self.beta = Tensor.fillWith(value: 0.0, size: .init(rows: 1, columns: inputSize.depth, depth: 1))
    }
  }
}


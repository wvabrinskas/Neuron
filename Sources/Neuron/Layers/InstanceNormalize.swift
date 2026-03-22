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
    let tensotContext = TensorContext { inputs, gradient, wrt in
      self.backward(inputs: inputs, gradient: gradient)
    }

    let forwardStorage = normalize(inputs: tensor)
    let out = Tensor(storage: forwardStorage, size: tensor.size, context: tensotContext)
    out.setGraph(tensor)

    return super.forward(tensor: out, context: context)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    setupTrainables()
  }

  private func normalize(inputs: Tensor) -> TensorStorage {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let pcSize = TensorSize(rows: 1, columns: 1, depth: depth)

    let means = TensorStorage.create(count: depth)
    for i in 0..<depth {
      means[i] = NumSwiftFlat.mean(inputs.depthPointer(i), count: sliceSize)
    }

    let meanT = Tensor(storage: means, size: pcSize)
    let centered = inputs - meanT

    let stds = TensorStorage.create(count: depth)
    
    for i in 0..<depth {
      let sumSq = NumSwiftFlat.sumOfSquares(centered.depthPointer(i), count: sliceSize)
      stds[i] = Tensor.Scalar.sqrt(sumSq / Tensor.Scalar(sliceSize) + epsilon)
    }

    let stdT = Tensor(storage: stds, size: pcSize)
    let gammaT = Tensor(storage: gamma.storage, size: pcSize)
    let betaT = Tensor(storage: beta.storage, size: pcSize)

    let normalized = centered / stdT
    let output = normalized * gammaT + betaT
    return output.storage
  }
  
  private func backward(inputs: Tensor, gradient: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let N = Tensor.Scalar(sliceSize)
    let pcSize = TensorSize(rows: 1, columns: 1, depth: depth)

    let means = TensorStorage.create(count: depth)
    let stds = TensorStorage.create(count: depth)

    for i in 0..<depth {
      means[i] = NumSwiftFlat.mean(inputs.depthPointer(i), count: sliceSize)
    }
    
    let meanT = Tensor(storage: means, size: pcSize)
    let centered = inputs - meanT

    for i in 0..<depth {
      let sumSq = NumSwiftFlat.sumOfSquares(centered.depthPointer(i), count: sliceSize)
      stds[i] = Tensor.Scalar.sqrt(sumSq / N + epsilon)
    }

    let stdT = Tensor(storage: stds, size: pcSize)
    let xNorm = centered / stdT

    let xNormTimesGrad = xNorm * gradient
    let dGammas = TensorStorage.create(count: depth)
    let dBetas = TensorStorage.create(count: depth)
    
    for i in 0..<depth {
      dBetas[i] = NumSwiftFlat.sum(gradient.depthPointer(i), count: sliceSize)
      dGammas[i] = NumSwiftFlat.sum(xNormTimesGrad.depthPointer(i), count: sliceSize)
    }

    let gammaT = Tensor(storage: gamma.storage, size: pcSize)
    let invNStdT = gammaT / (Tensor(N) * stdT)
    let dBetaT = Tensor(storage: dBetas, size: pcSize)
    let dGammaT = Tensor(storage: dGammas, size: pcSize)

    let dInput = (gradient * N - dBetaT - xNorm * dGammaT) * invNStdT

    let dGammaTensor = Tensor(storage: dGammas, size: TensorSize(rows: 1, columns: depth, depth: 1))
    let dBetaTensor  = Tensor(storage: dBetas, size: TensorSize(rows: 1, columns: depth, depth: 1))

    return (dInput,
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

    gamma = gamma.copy() - gammaWeights
    beta = beta.copy() - betaWeights
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


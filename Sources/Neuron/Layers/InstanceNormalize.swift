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
      self.backwardFlat(inputs: inputs, gradient: gradient)
    }

    let forwardStorage = normalizeFlat(inputs: tensor)
    let out = Tensor(storage: forwardStorage, size: tensor.size, context: tensotContext)
    out.setGraph(tensor)

    return super.forward(tensor: out, context: context)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    setupTrainables()
  }

  private func normalizeFlat(inputs: Tensor) -> TensorStorage {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let pcSize = TensorSize(rows: 1, columns: 1, depth: depth)

    var means = [Tensor.Scalar](repeating: 0, count: depth)
    var stds  = [Tensor.Scalar](repeating: 0, count: depth)

    let centeredBuf = TensorStorage.create(count: sliceSize)

    for i in 0..<depth {
      let inPtr = inputs.storage.pointer + i * sliceSize
      let mean = NumSwiftFlat.mean(inPtr, count: sliceSize)
      means[i] = mean

      NumSwiftFlat.sub(inPtr, scalar: mean, result: centeredBuf.pointer, count: sliceSize)
      let sumSq = NumSwiftFlat.sumOfSquares(centeredBuf.pointer, count: sliceSize)
      stds[i] = Tensor.Scalar.sqrt(sumSq / Tensor.Scalar(sliceSize) + epsilon)
    }

    let meanT  = Tensor(storage: TensorStorage.create(from: means), size: pcSize)
    let stdT   = Tensor(storage: TensorStorage.create(from: stds), size: pcSize)
    let gammaT = Tensor(storage: gamma.storage, size: pcSize)
    let betaT  = Tensor(storage: beta.storage, size: pcSize)

    let normalized = (inputs - meanT) / stdT
    let output = normalized * gammaT + betaT
    return output.storage
  }
  
  private func backwardFlat(inputs: Tensor, gradient: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let N = Tensor.Scalar(sliceSize)

    let dInputResult = TensorStorage.create(count: sliceSize * depth)
    let dGammaResult = TensorStorage.create(count: depth)
    let dBetaResult  = TensorStorage.create(count: depth)

    // Scratch buffers reused each depth slice
    let xNorm = TensorStorage.create(count: sliceSize)  // (input - mean) / std
    let tmp   = TensorStorage.create(count: sliceSize)  // general scratch

    for i in 0..<depth {
      let gammaVal = gamma.storage[i]
      let inPtr    = inputs.storage.pointer + i * sliceSize
      let gradPtr  = gradient.storage.pointer + i * sliceSize
      let outPtr   = dInputResult.pointer + i * sliceSize

      // mean, variance, std  (matching forward pass)
      let mean = NumSwiftFlat.mean(inPtr, count: sliceSize)
      NumSwiftFlat.sub(inPtr, scalar: mean, result: xNorm.pointer, count: sliceSize)
      let sumSq = NumSwiftFlat.sumOfSquares(xNorm.pointer, count: sliceSize)
      let std = Tensor.Scalar.sqrt(sumSq / N + epsilon)

      // x_norm = (input - mean) / std  (reuse xNorm buffer)
      NumSwiftFlat.div(xNorm.pointer, scalar: std, result: xNorm.pointer, count: sliceSize)

      // dL_dbeta = sum(grad)
      let dL_dbeta = NumSwiftFlat.sum(gradPtr, count: sliceSize)

      // dL_dgamma = sum(x_norm * grad)
      NumSwiftFlat.mul(xNorm.pointer, gradPtr, result: tmp.pointer, count: sliceSize)
      let dL_dgamma = NumSwiftFlat.sum(tmp.pointer, count: sliceSize)

      // invNStd = gamma[i] / (N * std)
      let invNStd = gammaVal / (N * std)

      // dl_dx = invNStd * (N * grad - dL_dbeta - x_norm * dL_dgamma)
      // tmp = N * grad - dL_dbeta
      NumSwiftFlat.mul(gradPtr, scalar: N, result: tmp.pointer, count: sliceSize)
      NumSwiftFlat.sub(tmp.pointer, scalar: dL_dbeta, result: tmp.pointer, count: sliceSize)
      // xNorm = x_norm * dL_dgamma  (reuse xNorm)
      NumSwiftFlat.mul(xNorm.pointer, scalar: dL_dgamma, result: xNorm.pointer, count: sliceSize)
      // tmp = tmp - xNorm
      NumSwiftFlat.sub(tmp.pointer, xNorm.pointer, result: tmp.pointer, count: sliceSize)
      // outPtr = invNStd * tmp
      NumSwiftFlat.mul(tmp.pointer, scalar: invNStd, result: outPtr, count: sliceSize)

      dGammaResult[i] = dL_dgamma
      dBetaResult[i]  = dL_dbeta
    }

    let dGammaTensor = Tensor(storage: dGammaResult, size: TensorSize(rows: 1, columns: depth, depth: 1))
    let dBetaTensor  = Tensor(storage: dBetaResult,  size: TensorSize(rows: 1, columns: depth, depth: 1))

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


//
//  GroupNormalize.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/6/26.
//

import Foundation
import NumSwift

/// A layer that performs group normalization on its input.
///
/// Group normalization splits the `C` channels of the input into `G` contiguous
/// groups of `C / G` channels each, and normalizes over all spatial positions
/// **and** all channels within a group (so `(C / G) · H · W` elements per statistic).
/// A learnable **per-channel** scale (gamma) and shift (beta) are then applied.
///
/// It is the general case between two layers already in the framework:
/// - `groups == 1`          ⇒ equivalent to `LayerNormalize`.
/// - `groups == C` (depth)  ⇒ equivalent to `InstanceNormalize`.
///
/// Unlike `BatchNormalize`, statistics never depend on the batch, so no cross-batch
/// synchronization is required and the layer normalizes each tensor independently.
public final class GroupNormalize: BaseLayer {
  /// The combined weights tensor of beta and gamma, used primarily for display and inspection purposes.
  /// - Returns: A `Tensor` containing the concatenated beta and gamma values.
  public override var weights: Tensor {
    get {
      // Must match gradient layout (beta | gamma) for correct optimizer weight decay indexing
      return beta.concat(gamma, axis: 2)
    }
    set {}
  }

  /// The number of groups the channels are partitioned into.
  public let groups: Int

  private var epsilon: Tensor.Scalar
  /// The gamma scaling parameter used in group normalization. Shape: `(1, C, 1)`.
  public var gamma: Tensor = .init()
  /// The beta shift parameter used in group normalization. Shape: `(1, C, 1)`.
  public var beta: Tensor = .init()

  /// Coding keys used for encoding and decoding the group normalization layer.
  public enum CodingKeys: String, CodingKey {
    /// Key for the learnable gamma scale parameter.
    case gamma
    /// Key for the learnable beta shift parameter.
    case beta
    /// Key for the epsilon stability term.
    case epsilon
    /// Key for the number of groups.
    case groups
    /// Key for the layer's input size.
    case inputSize
    /// Key for the layer's stable link identifier.
    case linkId
  }

  /// Default initializer for group normalization.
  /// - Parameters:
  ///   - groups: Number of groups to split the channels into. Must evenly divide the channel count.
  ///   - epsilon: Epsilon value for numerical stability. Default: `.stabilityFactor`
  ///   - gamma: Optional preloaded per-channel scale. Shape `(1, C, 1)`.
  ///   - beta: Optional preloaded per-channel shift. Shape `(1, C, 1)`.
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  public init(groups: Int,
              epsilon: Tensor.Scalar = .stabilityFactor,
              gamma: Tensor = .init(),
              beta: Tensor = .init(),
              inputSize: TensorSize? = nil,
              linkId: String = UUID().uuidString) {
    self.groups = groups
    self.epsilon = epsilon
    self.beta = beta
    self.gamma = gamma

    super.init(inputSize: inputSize,
               biasEnabled: false,
               linkId: linkId,
               encodingType: .groupNormalize)

    if let inputSize {
      self.outputSize = inputSize
    }

    setupTrainables()
  }

  /// Decodes a GroupNormalize layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let gamma = try container.decodeIfPresent(Tensor.self, forKey: .gamma) ?? .init()
    let beta = try container.decodeIfPresent(Tensor.self, forKey: .beta) ?? .init()
    let epsilon = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .epsilon) ?? .stabilityFactor
    let groups = try container.decodeIfPresent(Int.self, forKey: .groups) ?? 1
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString

    self.init(groups: groups,
              epsilon: epsilon,
              gamma: gamma,
              beta: beta,
              linkId: linkId)

    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
    self.outputSize = inputSize

    setupTrainables()
  }

  /// Encodes group-normalization parameters.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  /// - Throws: An error if any value fails to encode.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(epsilon, forKey: .epsilon)
    try container.encode(groups, forKey: .groups)
    try container.encode(linkId, forKey: .linkId)
  }

  /// Applies group normalization over each group of channels.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Normalized tensor with group-norm backpropagation context.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let tensorContext = TensorContext { inputs, gradient, wrt in
      self.backward(inputs: inputs, gradient: gradient)
    }

    let forwardStorage = normalize(inputs: tensor)
    let out = Tensor(storage: forwardStorage, size: tensor.size, context: tensorContext)
    out.setGraph(tensor)

    return super.forward(tensor: out, context: context)
  }

  /// Called by `Sequential.compile()` when input size is propagated; sets output size and initializes gamma/beta trainables.
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    precondition(inputSize.depth % groups == 0,
                 "GroupNormalize: channel count (\(inputSize.depth)) must be divisible by groups (\(groups)).")
    outputSize = inputSize
    setupTrainables()
  }

  // MARK: - Private

  /// Builds per-channel broadcast tensors of the group mean and group std.
  ///
  /// Every channel in group `g` receives the same `mean_g` / `std_g`, so a plain
  /// `(1, 1, C)` broadcast tensor lets us subtract/divide across the whole input
  /// with the framework's existing element-wise ops.
  private func groupStatistics(centeredOf inputs: Tensor,
                               sliceSize: Int,
                               channelsPerGroup: Int) -> (mean: Tensor, std: Tensor, centered: Tensor) {
    let depth = inputs.size.depth
    let m = Tensor.Scalar(channelsPerGroup * sliceSize)
    let pcSize = TensorSize(rows: 1, columns: 1, depth: depth)

    // --- group mean, broadcast to every channel in the group ---
    let means = TensorStorage.create(count: depth)
    for g in 0..<groups {
      let start = g * channelsPerGroup
      var sum: Tensor.Scalar = 0
      for c in start..<(start + channelsPerGroup) {
        sum += NumSwiftFlat.sum(inputs.depthPointer(c), count: sliceSize)
      }
      let mean = sum / m
      for c in start..<(start + channelsPerGroup) { means[c] = mean }
    }
    let meanT = Tensor(storage: means, size: pcSize)
    let centered = inputs - meanT

    // --- group std, broadcast to every channel in the group ---
    let stds = TensorStorage.create(count: depth)
    for g in 0..<groups {
      let start = g * channelsPerGroup
      var sumSq: Tensor.Scalar = 0
      for c in start..<(start + channelsPerGroup) {
        sumSq += NumSwiftFlat.sumOfSquares(centered.depthPointer(c), count: sliceSize)
      }
      let std = Tensor.Scalar.sqrt(sumSq / m + epsilon)
      for c in start..<(start + channelsPerGroup) { stds[c] = std }
    }
    let stdT = Tensor(storage: stds, size: pcSize)

    return (meanT, stdT, centered)
  }

  private func normalize(inputs: Tensor) -> TensorStorage {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let channelsPerGroup = depth / groups
    let pcSize = TensorSize(rows: 1, columns: 1, depth: depth)

    let stats = groupStatistics(centeredOf: inputs,
                                sliceSize: sliceSize,
                                channelsPerGroup: channelsPerGroup)

    let gammaT = Tensor(storage: gamma.storage, size: pcSize)
    let betaT = Tensor(storage: beta.storage, size: pcSize)

    let normalized = stats.centered / stats.std
    let output = normalized * gammaT + betaT
    return output.storage
  }

  private func backward(inputs: Tensor, gradient: Tensor) -> (input: Tensor, weight: Tensor, bias: Tensor) {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let channelsPerGroup = depth / groups
    let m = Tensor.Scalar(channelsPerGroup * sliceSize)
    let pcSize = TensorSize(rows: 1, columns: 1, depth: depth)

    let stats = groupStatistics(centeredOf: inputs,
                                sliceSize: sliceSize,
                                channelsPerGroup: channelsPerGroup)
    let stdT = stats.std
    let xNorm = stats.centered / stdT

    // --- parameter gradients (per channel) ---
    // dBeta_c  = Σ_i g_i          (over channel c's spatial slice)
    // dGamma_c = Σ_i g_i · x̂_i
    let xNormTimesGrad = xNorm * gradient
    let dGammas = TensorStorage.create(count: depth)
    let dBetas = TensorStorage.create(count: depth)
    for c in 0..<depth {
      dBetas[c] = NumSwiftFlat.sum(gradient.depthPointer(c), count: sliceSize)
      dGammas[c] = NumSwiftFlat.sum(xNormTimesGrad.depthPointer(c), count: sliceSize)
    }

    // --- input gradient ---
    // Fold per-channel gamma into dxhat BEFORE the group reduction, because gamma
    // varies across channels *within* a group (this is the key GroupNorm difference).
    let gammaT = Tensor(storage: gamma.storage, size: pcSize)
    let dxhat = gradient * gammaT                 // dx̂_i = g_i · γ_c
    let dxhatTimesXnorm = dxhat * xNorm

    // sum1_g = Σ_{i∈g} dx̂_i ,  sum2_g = Σ_{i∈g} dx̂_i · x̂_i  (broadcast per channel)
    let sum1 = TensorStorage.create(count: depth)
    let sum2 = TensorStorage.create(count: depth)
    for g in 0..<groups {
      let start = g * channelsPerGroup
      var s1: Tensor.Scalar = 0
      var s2: Tensor.Scalar = 0
      for c in start..<(start + channelsPerGroup) {
        s1 += NumSwiftFlat.sum(dxhat.depthPointer(c), count: sliceSize)
        s2 += NumSwiftFlat.sum(dxhatTimesXnorm.depthPointer(c), count: sliceSize)
      }
      for c in start..<(start + channelsPerGroup) { sum1[c] = s1; sum2[c] = s2 }
    }
    let sum1T = Tensor(storage: sum1, size: pcSize)
    let sum2T = Tensor(storage: sum2, size: pcSize)

    // dx_i = ( M·dx̂_i − sum1 − x̂_i·sum2 ) / (M·std)
    let dInput = (dxhat * m - sum1T - xNorm * sum2T) / (stdT * Tensor(m))

    // Pack weight grads as beta | gamma to match the `weights` layout.
    let dGammaTensor = Tensor(storage: dGammas, size: TensorSize(rows: 1, columns: depth, depth: 1))
    let dBetaTensor = Tensor(storage: dBetas, size: TensorSize(rows: 1, columns: depth, depth: 1))

    return (Tensor(storage: dInput.storage, size: inputs.size),
            dBetaTensor.concat(dGammaTensor, axis: 2),
            Tensor())
  }

  /// Applies group-normalization parameter updates.
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

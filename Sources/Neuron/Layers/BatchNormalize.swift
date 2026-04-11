//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/2/22.
//

import Foundation
import NumSwift

/// Standard per-channel batch normalization.
///
/// Computes mean and variance over both the batch **and** spatial dimensions
/// for each channel, yielding N×H×W samples per statistic. This is far more
/// stable than per-position normalization (which uses only N samples).
///
/// ## Thread Synchronization
///
/// The layer uses `BaseThreadBatchingLayer`'s two-phase pattern:
/// 1. **Phase 1 (accumulate):** Each worker calls `performThreadBatchingForwardPass`
///    under a lock, accumulating per-channel sums and sum-of-squares.
/// 2. **Barrier:** Workers wait until all batch items have been accumulated.
/// 3. **Phase 2 (normalize):** Each worker calls `forward(tensor:context:)` which
///    reads the fully-accumulated per-channel statistics to normalize its chunk.
///
/// Backpropagation uses the frozen-BN gradient `dx = dy × γ / σ`.
public final class BatchNormalize: BaseThreadBatchingLayer {
  /// Learnable scale parameters, one per channel. Shape: `(1, 1, depth)`.
  public var gamma: Tensor = .init()
  /// Learnable shift parameters, one per channel. Shape: `(1, 1, depth)`.
  public var beta: Tensor = .init()
  /// Per-channel moving mean for inference. Shape: `(1, 1, depth)`.
  public var movingMean: Tensor = .init()
  /// Per-channel moving variance for inference. Shape: `(1, 1, depth)`.
  public var movingVariance: Tensor = .init()

  /// Momentum for exponential moving average of inference statistics.
  public let momentum: Tensor.Scalar

  /// A combined tensor of all learnable and tracked parameters for this batch normalization layer.
  ///
  /// Returns a `Tensor` concatenating `beta`, `gamma`, `movingMean`, and `movingVariance` in order.
  /// Setting this property has no effect.
  public override var weights: Tensor {
    get {
      var combined = beta.storage.toArray()
      combined.append(contentsOf: gamma.storage.toArray())
      combined.append(contentsOf: movingMean.storage.toArray())
      combined.append(contentsOf: movingVariance.storage.toArray())
      return Tensor(combined)
    }
    set {}
  }

  /// Indicates whether the layer should accumulate batch statistics during the forward pass.
  ///
  /// Returns `true` when the layer is in training mode, enabling batch normalization accumulation.
  public override var shouldPerformBatching: Bool { isTraining }

  private let e: Tensor.Scalar = 1e-5
  private var dGamma: [Tensor.Scalar] = []
  private var dBeta: [Tensor.Scalar] = []

  // Per-channel batch accumulators (reset each optimization step)
  private var channelSums: [Tensor.Scalar] = []
  private var channelSumSqs: [Tensor.Scalar] = []
  internal private(set) var sampleCount: Int = 0
  private var scratchBuffer: TensorStorage = .create(count: 0)

  // Cached per-channel batch statistics (set during normalize, used in apply)
  private var batchMeans: [Tensor.Scalar] = []
  private var batchVariances: [Tensor.Scalar] = []

  private struct Normalization {
    let normalized: Tensor
    let stds: [Tensor.Scalar]
  }

  /// Creates a batch-normalization layer.
  ///
  /// - Parameters:
  ///   - gamma: Optional per-channel scale parameters.
  ///   - beta: Optional per-channel bias parameters.
  ///   - momentum: Moving-average momentum for inference statistics.
  ///   - movingMean: Optional preloaded per-channel moving means.
  ///   - movingVariance: Optional preloaded per-channel moving variances.
  ///   - inputSize: Optional input tensor shape.
  ///   - linkId: Unique identifier for graph linking.
  public init(gamma: [Tensor.Scalar] = [],
              beta: [Tensor.Scalar] = [],
              momentum: Tensor.Scalar = 0.99,
              movingMean: [Tensor.Scalar] = [],
              movingVariance: [Tensor.Scalar] = [],
              inputSize: TensorSize? = nil,
              linkId: String = UUID().uuidString) {
    self.gamma = Self.perChannelTensor(from: gamma)
    self.beta = Self.perChannelTensor(from: beta)
    self.movingMean = Self.perChannelTensor(from: movingMean)
    self.movingVariance = Self.perChannelTensor(from: movingVariance)
    self.momentum = momentum

    super.init(inputSize: inputSize,
               linkId: linkId,
               encodingType: .batchNormalize)

    self.usesOptimizer = false
    setupTrainables()
    resetDeltas()
    resetChannelAccumulators()
  }

  // MARK: - Codable

  /// Coding keys used to encode and decode the batch normalization layer's parameters.
  public enum CodingKeys: String, CodingKey {
    case gamma, beta, momentum, movingMean, movingVariance, inputSize, linkId
  }

  /// Decodes a BatchNormalize layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  convenience public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let gamma = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .gamma) ?? []
    let beta = try container.decodeIfPresent([Tensor.Scalar].self, forKey: .beta) ?? []
    let momentum = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .momentum) ?? 0.99

    let inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])

    // Decode moving stats — handle both new per-channel [Scalar] and legacy per-position Tensor
    let decodedMovingMean: [Tensor.Scalar] = try Self.decodeMovingStats(
      from: container, key: .movingMean, inputSize: inputSize)

    let decodedMovingVar: [Tensor.Scalar] = try Self.decodeMovingStats(
      from: container, key: .movingVariance, inputSize: inputSize)

    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString

    self.init(gamma: gamma,
              beta: beta,
              momentum: momentum,
              movingMean: decodedMovingMean,
              movingVariance: decodedMovingVar,
              linkId: linkId)

    self.inputSize = inputSize
    self.outputSize = inputSize
  }

  /// Encodes the layer's parameters into the given encoder.
  ///
  /// Encodes `inputSize`, `beta`, `gamma`, `momentum`, `movingMean`, `movingVariance`, and `linkId`.
  ///
  /// - Parameter encoder: The encoder to write layer parameters to.
  /// - Throws: An encoding error if any value cannot be encoded.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(beta.storage.toArray(), forKey: .beta)
    try container.encode(gamma.storage.toArray(), forKey: .gamma)
    try container.encode(momentum, forKey: .momentum)
    try container.encode(movingMean.storage.toArray(), forKey: .movingMean)
    try container.encode(movingVariance.storage.toArray(), forKey: .movingVariance)
    try container.encode(linkId, forKey: .linkId)
  }

  // MARK: - Forward

  /// Accumulates per-channel sums for one tensor during synchronized batching.
  public override func performThreadBatchingForwardPass(tensor: Tensor, context: NetworkContext) {
    guard isTraining else { return }
    let sliceSize = inputSize.rows * inputSize.columns
    updateLock.with {
      for c in 0..<tensor.size.depth {
        let ptr = tensor.depthPointer(c)
        channelSums[c] += NumSwiftFlat.sum(ptr, count: sliceSize)
        NumSwiftFlat.mul(ptr, ptr, result: scratchBuffer.pointer, count: sliceSize)
        channelSumSqs[c] += NumSwiftFlat.sum(scratchBuffer.pointer, count: sliceSize)
      }
      sampleCount += 1
    }
  }

  /// Applies batch normalization to an input tensor.
  ///
  /// Must be called **after** `forward(tensorBatch:context:)` has accumulated
  /// all batch members via `performThreadBatchingForwardPass`.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    let localIterations = iterations.load(ordering: .relaxed)

    let sizeToCheck = if context.totalInBatch != batchSize {
      context.totalInBatch
    } else {
      batchSize
    }

    if localIterations == 0 && context.totalInBatch == 1 {
      performThreadBatchingForwardPass(tensor: tensor, context: context)
    } else if context.totalInBatch > 1 && localIterations != sizeToCheck && isTraining {
      fatalError("Please call the tensorBatch function on BatchNorm with the total batch before calling this function \(#function)")
    }

    let forward = normalize(inputs: tensor, context: context)
    let normalization = forward.normalized

    let tensorContext = TensorContext { inputs, gradient, wrt in
      let backward = self.backward(inputs: inputs,
                                   gradient: gradient,
                                   normalization: normalization)
      backward.label = "BatchNorm"
      return (backward, Tensor(), Tensor())
    }

    let out = Tensor(storage: forward.output, size: tensor.size, context: tensorContext)
    out.setGraph(tensor)
    return super.forward(tensor: out, context: context)
  }

  // MARK: - Apply

  /// Updates gamma/beta, moving stats, then resets batch accumulators.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    super.apply(gradients: gradients, learningRate: learningRate)

    let N = Tensor.Scalar(max(sampleCount, 1))
    let avgDGammaT = Self.perChannelTensor(from: dGamma) / N
    let avgDBetaT = Self.perChannelTensor(from: dBeta) / N

    gamma = gamma.copy() - avgDGammaT * learningRate
    beta = beta.copy() - avgDBetaT * learningRate

    let batchMeansT = Self.perChannelTensor(from: batchMeans)
    let batchVariancesT = Self.perChannelTensor(from: batchVariances)
    movingMean = movingMean.copy() * momentum + batchMeansT * (1 - momentum)
    movingVariance = movingVariance.copy() * momentum + batchVariancesT * (1 - momentum)

    resetDeltas()
    resetChannelAccumulators()
  }

  // MARK: - Input Size

  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
    setupTrainables()
    resetDeltas()
    resetChannelAccumulators()
    let sliceSize = inputSize.rows * inputSize.columns
    if sliceSize > 0 {
      scratchBuffer = .create(count: sliceSize)
    }
  }

  // MARK: - Private

  private func setupTrainables() {
    let depth = inputSize.depth
    let pcSize = TensorSize(rows: 1, columns: 1, depth: depth)
    if gamma.isEmpty { gamma = Tensor(storage: TensorStorage.create(repeating: 1, count: depth), size: pcSize) }
    if beta.isEmpty { beta = Tensor(storage: TensorStorage.create(count: depth), size: pcSize) }
    if movingMean.isEmpty { movingMean = Tensor(storage: TensorStorage.create(count: depth), size: pcSize) }
    if movingVariance.isEmpty { movingVariance = Tensor(storage: TensorStorage.create(repeating: 1, count: depth), size: pcSize) }
    if batchMeans.count != depth { batchMeans = [Tensor.Scalar](repeating: 0, count: depth) }
    if batchVariances.count != depth { batchVariances = [Tensor.Scalar](repeating: 0, count: depth) }
  }

  private func resetDeltas() {
    let depth = inputSize.depth
    dGamma = [Tensor.Scalar](repeating: 0, count: depth)
    dBeta = [Tensor.Scalar](repeating: 0, count: depth)
  }

  private func resetChannelAccumulators() {
    let depth = inputSize.depth
    channelSums = [Tensor.Scalar](repeating: 0, count: depth)
    channelSumSqs = [Tensor.Scalar](repeating: 0, count: depth)
    sampleCount = 0
  }

  private func normalize(inputs: Tensor,
                         context: NetworkContext) -> (output: TensorStorage, normalized: Normalization) {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns

    let meanT: Tensor
    let stdT: Tensor
    var stds = [Tensor.Scalar](repeating: 0, count: depth)

    if isTraining {
      var means = [Tensor.Scalar](repeating: 0, count: depth)

      for c in 0..<depth {
        let M = Tensor.Scalar(max(sampleCount * sliceSize, 1))
        let mean_c = channelSums[c] / M
        let var_c = max(channelSumSqs[c] / M - mean_c * mean_c, 0)
        means[c] = mean_c
        stds[c] = Tensor.Scalar.sqrt(var_c + e)
        batchMeans[c] = mean_c
        batchVariances[c] = var_c
      }

      meanT = Self.perChannelTensor(from: means)
      stdT = Self.perChannelTensor(from: stds)
    } else {
      meanT = movingMean
      stdT = movingVariance.sqrt(adding: e)
      for c in 0..<depth { stds[c] = stdT.storage[c] }
    }

    let normalized = (inputs - meanT) / stdT
    let output = normalized * gamma + beta

    return (output.storage, Normalization(normalized: normalized, stds: stds))
  }

  private func backward(inputs: Tensor,
                        gradient: Tensor,
                        normalization: Normalization) -> Tensor {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns

    let gradTimesNorm = gradient * normalization.normalized

    for c in 0..<depth {
      updateLock.with {
        dGamma[c] += NumSwiftFlat.sum(gradTimesNorm.depthPointer(c), count: sliceSize)
        dBeta[c] += NumSwiftFlat.sum(gradient.depthPointer(c), count: sliceSize)
      }
    }

    let stdsT = Self.perChannelTensor(from: normalization.stds)
    let scaleT = gamma / stdsT
    return gradient * scaleT
  }

  private static func perChannelTensor(from scalars: [Tensor.Scalar]) -> Tensor {
    let depth = scalars.count
    guard depth > 0 else { return Tensor() }
    return Tensor(storage: TensorStorage.create(from: scalars),
                  size: TensorSize(rows: 1, columns: 1, depth: depth))
  }

  // MARK: - Codable Helpers

  /// Decodes per-channel moving statistics, handling legacy per-position Tensor formats
  /// by averaging over spatial dimensions.
  private static func decodeMovingStats(
    from container: KeyedDecodingContainer<CodingKeys>,
    key: CodingKeys,
    inputSize: TensorSize
  ) throws -> [Tensor.Scalar] {
    // New format: plain [Scalar] array with depth elements
    if let scalars = try? container.decodeIfPresent([Tensor.Scalar].self, forKey: key),
       scalars.count == inputSize.depth {
      return scalars
    }

    let sliceSize = inputSize.rows * inputSize.columns

    // Legacy: full Tensor with per-position stats — average over spatial dims
    if let tensor = try? container.decodeIfPresent(Tensor.self, forKey: key), !tensor.isEmpty {
      return Self.spatialAveragePerChannel(tensor, depth: inputSize.depth, sliceSize: sliceSize)
    }
    if let data = try? container.decodeIfPresent(Tensor.Data.self, forKey: key) {
      let tensor = Tensor(data)
      return Self.spatialAveragePerChannel(tensor, depth: inputSize.depth, sliceSize: sliceSize)
    }

    // Legacy: per-channel scalars broadcast to spatial — just use the scalars directly
    if let scalars = try? container.decodeIfPresent([Tensor.Scalar].self, forKey: key), !scalars.isEmpty {
      return scalars
    }

    return []
  }

  private static func spatialAveragePerChannel(_ tensor: Tensor,
                                               depth: Int,
                                               sliceSize: Int) -> [Tensor.Scalar] {
    guard sliceSize > 0 else { return [] }
    var result = [Tensor.Scalar](repeating: 0, count: depth)
    for c in 0..<depth {
      let ptr = tensor.storage.pointer + c * sliceSize
      result[c] = NumSwiftFlat.sum(ptr, count: sliceSize) / Tensor.Scalar(sliceSize)
    }
    return result
  }
}

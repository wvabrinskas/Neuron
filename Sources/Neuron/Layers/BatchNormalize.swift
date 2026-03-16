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
  /// Learnable scale parameters, one per channel.
  public var gamma: [Tensor.Scalar] = []
  /// Learnable shift parameters, one per channel.
  public var beta: [Tensor.Scalar] = []
  /// Per-channel moving mean for inference.
  public var movingMean: [Tensor.Scalar] = []
  /// Per-channel moving variance for inference.
  public var movingVariance: [Tensor.Scalar] = []

  /// Momentum for exponential moving average of inference statistics.
  public let momentum: Tensor.Scalar

  /// A combined tensor of all learnable and tracked parameters for this batch normalization layer.
  ///
  /// Returns a `Tensor` concatenating `beta`, `gamma`, `movingMean`, and `movingVariance` in order.
  /// Setting this property has no effect.
  public override var weights: Tensor {
    get {
      var combined = beta
      combined.append(contentsOf: gamma)
      combined.append(contentsOf: movingMean)
      combined.append(contentsOf: movingVariance)
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
    let normalized: TensorStorage
    let std: Tensor.Scalar
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
    self.gamma = gamma
    self.beta = beta
    self.movingMean = movingMean
    self.movingVariance = movingVariance
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

  /// Creates a `BatchNormalize` layer by decoding its parameters from the given decoder.
  ///
  /// Supports decoding both current per-channel scalar arrays and legacy per-position `Tensor` formats
  /// for moving mean and variance.
  ///
  /// - Parameter decoder: The decoder to read layer parameters from.
  /// - Throws: A decoding error if required values cannot be read or are in an unexpected format.
  public enum CodingKeys: String, CodingKey {
    case gamma, beta, momentum, movingMean, movingVariance, inputSize, linkId
  }

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
    try container.encode(beta, forKey: .beta)
    try container.encode(gamma, forKey: .gamma)
    try container.encode(momentum, forKey: .momentum)
    try container.encode(movingMean, forKey: .movingMean)
    try container.encode(movingVariance, forKey: .movingVariance)
    try container.encode(linkId, forKey: .linkId)
  }

  // MARK: - Forward

  /// Accumulates per-channel sums for one tensor during synchronized batching.
  public override func performThreadBatchingForwardPass(tensor: Tensor, context: NetworkContext) {
    guard isTraining else { return }
    let sliceSize = inputSize.rows * inputSize.columns
    updateLock.with {
      for c in 0..<tensor.size.depth {
        let ptr = tensor.storage.pointer + c * sliceSize
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
    let normalizations = forward.normalized

    let tensorContext = TensorContext { inputs, gradient, wrt in
      let backward = self.backward(inputs: inputs,
                                   gradient: gradient,
                                   normalizations: normalizations)
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
    let avgDGamma = dGamma / N
    let avgDBeta = dBeta / N

    gamma = gamma - (avgDGamma * learningRate)
    beta = beta - (avgDBeta * learningRate)

    // Update moving stats from cached batch statistics
    for c in 0..<inputSize.depth {
      movingMean[c] = momentum * movingMean[c] + (1 - momentum) * batchMeans[c]
      movingVariance[c] = momentum * movingVariance[c] + (1 - momentum) * batchVariances[c]
    }

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
    if gamma.isEmpty { gamma = [Tensor.Scalar](repeating: 1, count: depth) }
    if beta.isEmpty { beta = [Tensor.Scalar](repeating: 0, count: depth) }
    if movingMean.isEmpty { movingMean = [Tensor.Scalar](repeating: 0, count: depth) }
    if movingVariance.isEmpty { movingVariance = [Tensor.Scalar](repeating: 1, count: depth) }
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
                         context: NetworkContext) -> (output: TensorStorage, normalized: [Normalization]) {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let outStorage = TensorStorage.create(count: inputs.storage.count)
    var normalizedInputs: [Normalization] = []

    for c in 0..<depth {
      let inputPtr = inputs.storage.pointer + c * sliceSize
      let outPtr   = outStorage.pointer + c * sliceSize

      let mean_c: Tensor.Scalar
      let std_c: Tensor.Scalar

      if isTraining {
        let M = Tensor.Scalar(max(sampleCount * sliceSize, 1))
        mean_c = channelSums[c] / M
        let var_c = max(channelSumSqs[c] / M - mean_c * mean_c, 0)
        std_c = Tensor.Scalar.sqrt(var_c + e)

        batchMeans[c] = mean_c
        batchVariances[c] = var_c
      } else {
        mean_c = movingMean[c]
        std_c = Tensor.Scalar.sqrt(movingVariance[c] + e)
      }

      // normalized = (input - mean_c) / std_c
      let normStorage = TensorStorage.create(count: sliceSize)
      NumSwiftFlat.sub(inputPtr, scalar: mean_c, result: normStorage.pointer, count: sliceSize)
      NumSwiftFlat.div(normStorage.pointer, scalar: std_c, result: normStorage.pointer, count: sliceSize)

      normalizedInputs.append(Normalization(normalized: normStorage, std: std_c))

      // output = gamma_c * normalized + beta_c
      NumSwiftFlat.mul(normStorage.pointer, scalar: gamma[c], result: outPtr, count: sliceSize)
      NumSwiftFlat.add(outPtr, scalar: beta[c], result: outPtr, count: sliceSize)
    }

    return (outStorage, normalizedInputs)
  }

  private func backward(inputs: Tensor,
                        gradient: Tensor,
                        normalizations: [Normalization]) -> Tensor {
    let depth = inputs.size.depth
    let sliceSize = inputSize.rows * inputSize.columns
    let outStorage = TensorStorage.create(count: inputs.storage.count)
    let tmpA = TensorStorage.create(count: sliceSize)

    for c in 0..<depth {
      let gradPtr = gradient.storage.pointer + c * sliceSize
      let outPtr  = outStorage.pointer + c * sliceSize

      let normStorage = normalizations[c].normalized
      let std_c = normalizations[c].std

      updateLock.with {
        NumSwiftFlat.mul(gradPtr, normStorage.pointer, result: tmpA.pointer, count: sliceSize)
        dGamma[c] += NumSwiftFlat.sum(tmpA.pointer, count: sliceSize)
        dBeta[c] += NumSwiftFlat.sum(gradPtr, count: sliceSize)
      }

      // Frozen BN gradient: dx = dy * gamma_c / std_c
      let scale = gamma[c] / std_c
      NumSwiftFlat.mul(gradPtr, scalar: scale, result: outPtr, count: sliceSize)
    }

    return Tensor(storage: outStorage, size: inputs.size)
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

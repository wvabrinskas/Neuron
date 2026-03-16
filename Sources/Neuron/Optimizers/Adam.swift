//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// Convienence class for Adam with weight decay as default to `0.004`
public final class AdamW: Adam {
  /// Creates an AdamW optimizer with decoupled weight decay enabled.
  ///
  /// - Parameters:
  ///   - trainable: Model whose parameters will be optimized.
  ///   - device: Execution device for forward/backward math.
  ///   - learningRate: Base learning rate.
  ///   - batchSize: Number of samples per optimization step.
  ///   - b1: Exponential decay for first-moment estimates.
  ///   - b2: Exponential decay for second-moment estimates.
  ///   - eps: Numerical stability epsilon.
  ///   - weightDecayValue: Decoupled weight decay coefficient.
  ///   - weightClip: Optional weight clipping threshold.
  ///   - gradientClip: Optional gradient clipping threshold.
  ///   - augmenter: Optional training-time data augmenter.
  public init(_ trainable: Trainable,
              learningRate: Tensor.Scalar,
              batchSize: Int,
              b1: Tensor.Scalar = 0.9,
              b2: Tensor.Scalar = 0.999,
              eps: Tensor.Scalar = .stabilityFactor,
              weightDecayValue: Tensor.Scalar = 0.004,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil,
              augmenter: Augmenter? = nil) {
    super.init(trainable,
               learningRate: learningRate,
               batchSize: batchSize,
               b1: b1,
               b2: b2,
               eps: eps,
               weightDecay: .decay(weightDecayValue),
               weightClip: weightClip,
               gradientClip: gradientClip,
               augmenter: augmenter)
  }
}

/// An optimizer that implements the Adam (Adaptive Moment Estimation) algorithm.
/// Combines the benefits of momentum-based and adaptive learning rate methods
/// to efficiently update trainable parameters during training.
public class Adam: BaseOptimizer {
  /// Defines the weight decay strategy applied during parameter updates.
  ///
  /// - `none`: No weight decay is applied.
  /// - `decay`: Applies L2 weight decay with the specified scalar coefficient.
  public enum WeightDecay {
    case none
    case decay(Tensor.Scalar)
  }
  
  /// The trainable model whose parameters will be optimized.
  /// Setting this property triggers a rebuild of the optimizer's internal state
  /// to match the new model's parameter shapes.
  public override var trainable: Trainable {
    didSet {
      build()
    }
  }
  
  private var b1: Tensor.Scalar = 0.9
  private var b2: Tensor.Scalar = 0.999
  private var eps: Tensor.Scalar = .stabilityFactor
  
  private var m: [TensorStorage] = []
  private var v: [TensorStorage] = []
  private var vb: [TensorStorage] = []
  private var mb: [TensorStorage] = []
  private var t: Tensor.Scalar = 1
  private let weightDecay: WeightDecay
  
  /// Creates an Adam optimizer.
  ///
  /// - Parameters:
  ///   - trainable: Model whose parameters will be optimized.
  ///   - device: Execution device for forward/backward math.
  ///   - learningRate: Base learning rate.
  ///   - batchSize: Number of samples per optimization step.
  ///   - b1: Exponential decay for first-moment estimates.
  ///   - b2: Exponential decay for second-moment estimates.
  ///   - eps: Numerical stability epsilon.
  ///   - weightDecay: Optional decoupled weight decay configuration.
  ///   - weightClip: Optional weight clipping threshold.
  ///   - gradientClip: Optional gradient clipping threshold.
  ///   - augmenter: Optional training-time data augmenter.
  public init(_ trainable: Trainable,
              learningRate: Tensor.Scalar,
              batchSize: Int,
              b1: Tensor.Scalar = 0.9,
              b2: Tensor.Scalar = 0.999,
              eps: Tensor.Scalar = .stabilityFactor,
              weightDecay: WeightDecay = .none,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil,
              augmenter: Augmenter? = nil) {
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.weightDecay = weightDecay
    super.init(trainable: trainable,
               learningRate: learningRate,
               batchSize: batchSize,
               weightClip: weightClip,
               gradientClip: gradientClip,
               augmenter: augmenter)
    build()
  }
  
  /// Applies one Adam optimization step from accumulated gradients.
  public override func step() {
    var gradients = gradientAccumulator.accumulate()
    
    if let clip = gradientClip {
      gradients = gradients.gradientL2NormClip(clip, metrics: metricsReporter)
    }
    
    for i in 0..<trainable.layers.count {
      let layer = trainable.layers[i]
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
      
      var adamGradient: Gradient = (gradient, biasGradient)
      
      // only apply optimizer gradient if the layer is trainable by the optimizer
      if layer.trainable, layer.usesOptimizer {
        adamGradient = run(gradient: gradient, biasGradient: biasGradient, index: i, weights: layer.weights, bias: layer.biases)
      }
            
      layer.apply(gradients: adamGradient, learningRate: learningRate)
      
      weightClip(layer: layer)
    }
    
    t += 1
    
    super.step()
  }
  
  private func build() {
    m = [TensorStorage].init(repeating: TensorStorage.create(count: 0), count: trainable.layers.count)
    v = [TensorStorage].init(repeating: TensorStorage.create(count: 0), count: trainable.layers.count)
    vb = [TensorStorage].init(repeating: TensorStorage.create(count: 0), count: trainable.layers.count)
    mb = [TensorStorage].init(repeating: TensorStorage.create(count: 0), count: trainable.layers.count)
    trainable.compile()
  }
  
  private func run(gradient: Tensor, biasGradient: Tensor, index: Int, weights: Tensor, bias: Tensor) -> Optimizer.Gradient {
    let i = index
    let gradCount = gradient.storage.count

    if m[i].isEmpty || v[i].isEmpty {
      m[i] = TensorStorage.create(count: gradCount)
      v[i] = TensorStorage.create(count: gradCount)
    }
    
    let result = apply(m: &m[i],
                       v: &v[i],
                       gradient: gradient.storage,
                       decay: true,
                       weights: weights.storage,
                       size: gradient.size)

    let biasCount = biasGradient.storage.count

    if mb[i].isEmpty || vb[i].isEmpty {
      mb[i] = TensorStorage.create(count: biasCount)
      vb[i] = TensorStorage.create(count: biasCount)
    }

    let biases = apply(m: &mb[i],
                       v: &vb[i],
                       gradient: biasGradient.storage,
                       weights: bias.storage,
                       size: biasGradient.size)
    
    return (Tensor(storage: result, size: gradient.size), Tensor(storage: biases, size: biasGradient.size))
  }

  private func apply(m: inout TensorStorage,
                     v: inout TensorStorage,
                     gradient: TensorStorage,
                     decay: Bool = false,
                     weights: TensorStorage,
                     size: TensorSize) -> TensorStorage {

    // Hoist loop-invariant computations
    let oneMinusB1 = 1 - b1
    let oneMinusB2 = 1 - b2
    let mCorrectionFactor = 1 / (1 - Tensor.Scalar.pow(b1, Tensor.Scalar(t)))
    let vCorrectionFactor = 1 / (1 - Tensor.Scalar.pow(b2, Tensor.Scalar(t)))
    let shouldDecay: Tensor.Scalar? = decay ? {
      if case .decay(let d) = weightDecay { return d }
      return nil
    }() : nil

    let count = gradient.count
    let result = TensorStorage.create(count: count)

    for i in 0..<count {
      let g = gradient[i]
      m[i] = b1 * m[i] + oneMinusB1 * g
      v[i] = b2 * v[i] + oneMinusB2 * (g * g)

      let mHat = m[i] * mCorrectionFactor
      let vHat = v[i] * vCorrectionFactor

      var delta = learningRate * (mHat / (sqrt(vHat + eps)))

      if let decayValue = shouldDecay, i < weights.count {
        let decayLR = learningRate * decayValue * weights[i]
        delta += decayLR
      }

      result[i] = delta
    }

    return result
  }
  
  /// Resets Adam moment buffers and step counters.
  public override func reset() {
    t = 1
    m.removeAll(keepingCapacity: true)
    v.removeAll(keepingCapacity: true)
    mb.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }
}

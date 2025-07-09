//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// AdamW optimizer - Adam with weight decay
/// Convenience class for Adam optimizer with weight decay enabled by default
/// Weight decay helps prevent overfitting by penalizing large weights
public final class AdamW: Adam {
  /// Initializes AdamW optimizer with weight decay
  /// - Parameters:
  ///   - trainable: The model to optimize
  ///   - device: Computation device (CPU/GPU)
  ///   - learningRate: Learning rate for parameter updates
  ///   - b1: Exponential decay rate for first moment estimates. Default: 0.9
  ///   - b2: Exponential decay rate for second moment estimates. Default: 0.999
  ///   - eps: Small constant for numerical stability. Default: .stabilityFactor
  ///   - weightDecayValue: Weight decay coefficient. Default: 0.004
  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              b1: Tensor.Scalar = 0.9,
              b2: Tensor.Scalar = 0.999,
              eps: Tensor.Scalar = .stabilityFactor,
              weightDecayValue: Tensor.Scalar = 0.004) {
    super.init(trainable,
               device: device,
               learningRate: learningRate,
               b1: b1,
               b2: b2,
               eps: eps,
               weightDecay: .decay(weightDecayValue))
  }
}

/// Adam (Adaptive Moment Estimation) optimizer
/// Combines the advantages of AdaGrad and RMSprop optimizers
/// Maintains per-parameter learning rates adapted based on first and second moments
/// Generally works well for most deep learning tasks
public class Adam: BaseOptimizer {
  /// Weight decay options for regularization
  public enum WeightDecay {
    /// No weight decay applied
    case none
    /// Apply weight decay with specified coefficient
    case decay(Tensor.Scalar)
  }
  
  /// The trainable model being optimized
  /// When set, rebuilds internal state for the new model
  public override var trainable: Trainable {
    didSet {
      build()
    }
  }
  
  /// Exponential decay rate for first moment estimates (momentum)
  private var b1: Tensor.Scalar = 0.9
  /// Exponential decay rate for second moment estimates (variance)
  private var b2: Tensor.Scalar = 0.999
  /// Small constant for numerical stability
  private var eps: Tensor.Scalar = .stabilityFactor
  
  /// First moment estimates for weights
  private var m: [Tensor.Data] = []
  /// Second moment estimates for weights
  private var v: [Tensor.Data] = []
  /// Second moment estimates for biases
  private var vb: [[Tensor.Scalar]] = []
  /// First moment estimates for biases
  private var mb: [[Tensor.Scalar]] = []
  /// Time step counter for bias correction
  private var t: Tensor.Scalar = 1
  /// Weight decay configuration
  private let weightDecay: WeightDecay
  
  /// Initializes Adam optimizer with specified parameters
  /// - Parameters:
  ///   - trainable: The model to optimize
  ///   - device: Computation device (CPU/GPU). Default: CPU()
  ///   - learningRate: Learning rate for parameter updates
  ///   - b1: Exponential decay rate for first moment estimates. Default: 0.9
  ///   - b2: Exponential decay rate for second moment estimates. Default: 0.999
  ///   - eps: Small constant for numerical stability. Default: .stabilityFactor
  ///   - weightDecay: Weight decay configuration. Default: .none
  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              b1: Tensor.Scalar = 0.9,
              b2: Tensor.Scalar = 0.999,
              eps: Tensor.Scalar = .stabilityFactor,
              weightDecay: WeightDecay = .none) {
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.weightDecay = weightDecay
    super.init(trainable: trainable,
               learningRate: learningRate,
               l2Normalize: false)
    build()
  }
  
  /// Performs one optimization step using Adam algorithm
  /// Updates model parameters using accumulated gradients and moment estimates
  /// Applies bias correction and optional weight decay
  public override func step() {
    let gradients = gradientAccumulator.accumulate()
    
    for i in 0..<trainable.layers.count {
      let layer = trainable.layers[i]
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
      
      if l2Normalize {
        gradient.l2Normalize()
      }
      
      var adamGradient: Gradient = (gradient, biasGradient)
      
      // only apply optimizer gradient if the layer is trainable by the optimizer
      if layer.trainable, layer.usesOptimizer {
        adamGradient = run(gradient: gradient, biasGradient: biasGradient, index: i, weights: layer.weights)
      }
      
      layer.apply(gradients: adamGradient, learningRate: learningRate)
      
      clip(layer: layer)
    }
    
    t += 1
    
    super.step()
  }
  
  /// Builds internal state arrays for moment estimates
  /// Initializes first and second moment estimates for all layers
  /// Called when trainable model is set or changed
  private func build() {
    m = [Tensor.Data].init(repeating: [], count: trainable.layers.count) // we want to support multiple weight structures right now this only supports one Tensor for one m value, when layers could have multiple tensors representing weights
    v = [Tensor.Data].init(repeating: [], count: trainable.layers.count)
    vb = [[Tensor.Scalar]].init(repeating: [], count: trainable.layers.count)
    mb = [[Tensor.Scalar]].init(repeating: [], count: trainable.layers.count)
    trainable.compile()
  }
  
  private func run(gradient: Tensor, biasGradient: Tensor, index: Int, weights: Tensor) -> Optimizer.Gradient {
    let shape = gradient.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    let depth = shape[safe: 2] ?? 0
    let i = index
    
    let flatBias = biasGradient.value.flatten()

    var result: [[[Tensor.Scalar]]] = []
    var biases: [Tensor.Scalar] = .init(repeating: 0, count: flatBias.count)

    if m[i].isEmpty || v[i].isEmpty {
      m[i] = NumSwift.zerosLike((rows, columns, depth))
      v[i] = NumSwift.zerosLike((rows, columns, depth))
      mb[i] = [Tensor.Scalar].init(repeating: 0, count: flatBias.count)
      vb[i] = [Tensor.Scalar].init(repeating: 0, count: flatBias.count)
    }
    
    let gradientValue = gradient.value
        
    for d in 0..<gradientValue.count {
      let depthGradient = gradientValue[d]
      var row: [[Tensor.Scalar]] = []
      for r in 0..<depthGradient.count {
        let rowGradient = depthGradient[r]
        var column: [Tensor.Scalar] = []
        for c in 0..<rowGradient.count {
          m[i][d][r][c] = b1 * m[i][d][r][c] + (1 - b1) * gradientValue[d][r][c]
          v[i][d][r][c] = b2 * v[i][d][r][c] + (1 - b2) * Tensor.Scalar.pow(gradientValue[d][r][c], 2)
          
          let mHat = m[i][d][r][c] / (1 - Tensor.Scalar.pow(b1, Tensor.Scalar(t)))
          let vHat = v[i][d][r][c] / (1 - Tensor.Scalar.pow(b2, Tensor.Scalar(t)))
          
          var delta = learningRate * (mHat / (sqrt(vHat + eps)))
          
          if case .decay(let decay) = weightDecay {
            let decay = learningRate * decay * weights.value[safe: d, [[]]][safe: r, []][safe: c, 1]
            delta -= decay
          }
          
          column.append(delta)
        }
        row.append(column)
      }
      
      result.append(row)
    }
    
    for d in 0..<flatBias.count {
      // bias gradients are performed at a depth level
      let gradientSum = flatBias[d]
      mb[i][d] = b1 * mb[i][d] + (1 - b1) * gradientSum
      vb[i][d] = b2 * vb[i][d] + (1 - b2) * Tensor.Scalar.pow(gradientSum, 2)
      
      let mHat = mb[i][d] / (1 - Tensor.Scalar.pow(b1, Tensor.Scalar(t)))
      let vHat = vb[i][d] / (1 - Tensor.Scalar.pow(b2, Tensor.Scalar(t)))
      
      let deltaB = learningRate * (mHat / (sqrt(vHat + eps)))
      
      biases[d] = deltaB
    }
        
    return (Tensor(result), Tensor(biases))
  }
  
  /// Resets the optimizer state to initial conditions
  /// Clears moment estimates and resets time step counter
  /// Useful when starting training from scratch
  public override func reset() {
    t = 1
    m.removeAll(keepingCapacity: true)
    v.removeAll(keepingCapacity: true)
    mb.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }
}

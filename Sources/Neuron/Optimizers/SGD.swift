//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// Stochastic Gradient Descent (SGD) optimizer with momentum
/// Classic optimization algorithm that updates parameters in the direction of negative gradient
/// Momentum helps accelerate gradients in relevant directions and dampens oscillations
/// Simple and effective for many machine learning tasks
public class SGD: BaseOptimizer {
  /// Momentum factor for accumulating gradients over time
  private let momentum: Tensor.Scalar
  /// Momentum vectors for weight updates
  private var v: [Tensor.Data] = []
  /// Momentum vectors for bias updates
  private var vb: [[Tensor.Scalar]] = []

  /// Initializes SGD optimizer with momentum
  /// - Parameters:
  ///   - trainable: The model to optimize
  ///   - device: Computation device (CPU/GPU). Default: CPU()
  ///   - learningRate: Learning rate for parameter updates
  ///   - momentum: Momentum factor for gradient accumulation. Default: 0.9
  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              momentum: Tensor.Scalar = 0.9) {
    self.momentum = momentum
    
    trainable.compile()
    v = [[[[Tensor.Scalar]]]].init(repeating: [], count: trainable.layers.count)
    vb = [[Tensor.Scalar]].init(repeating: [], count: trainable.layers.count)
    
    super.init(trainable: trainable,
               learningRate: learningRate,
               l2Normalize: false)
  }
  
  /// Performs one optimization step using SGD with momentum
  /// Updates parameters using momentum-adjusted gradients
  /// Formula: v = momentum * v + learning_rate * gradient, params -= v
  public override func step() {
    let gradients = gradientAccumulator.accumulate()

    for i in 0..<trainable.layers.count {
      let layer = trainable.layers[i]
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
      
      if l2Normalize {
        gradient.l2Normalize()
      }
      
      var sgdGradient = (gradient, biasGradient)
      
      if layer.trainable, layer.usesOptimizer {
        sgdGradient = run(gradient: gradient, biasGradient: biasGradient, index: i)
      }
      
      layer.apply(gradients: sgdGradient, learningRate: learningRate)
      
      clip(layer: layer)
    }
  }
  
  private func run(gradient: Tensor, biasGradient: Tensor, index: Int) -> Optimizer.Gradient {
    let shape = gradient.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    let depth = shape[safe: 2] ?? 0
    let i = index
    
    let flatBias = biasGradient.value.flatten()

    if v[i].isEmpty {
      v[i] = NumSwift.zerosLike((rows, columns, depth))
      vb[i] = [Tensor.Scalar].init(repeating: 0, count: flatBias.count)
    }
    
    let gradientValue = gradient.value
    
    for d in 0..<gradientValue.count {
      let depthGradient = gradientValue[d]
      for r in 0..<depthGradient.count {
        let rowGradient = depthGradient[r]
        for c in 0..<rowGradient.count {
          v[i][d][r][c] = momentum * v[i][d][r][c] + learningRate * gradientValue[d][r][c]
        }
      }
      
    }
          
    for d in 0..<flatBias.count {
      vb[i][d] = momentum * vb[i][d] + learningRate * flatBias[d]
    }
    
    return (Tensor(v[i]), Tensor(vb[i]))
  }
  
  /// Resets the optimizer state to initial conditions
  /// Clears momentum vectors for both weights and biases
  /// Useful when restarting training or changing models
  public override func reset() {
    v.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }
}

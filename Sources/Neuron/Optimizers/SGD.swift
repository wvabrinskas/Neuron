//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// Stochastic Gradient Descent optimizer with optional momentum and gradient/weight clipping support.
public class SGD: BaseOptimizer {
  private let momentum: Tensor.Scalar
  private var v: [TensorStorage] = []
  private var vb: [TensorStorage] = []

  /// Creates an SGD optimizer with optional momentum and clipping.
  ///
  /// - Parameters:
  ///   - trainable: Model whose parameters will be optimized.
  ///   - device: Execution device for forward/backward math.
  ///   - learningRate: Step size applied to parameter updates.
  ///   - batchSize: Number of samples per optimization step.
  ///   - momentum: Momentum coefficient applied to gradient velocity.
  ///   - weightClip: Optional weight clipping threshold.
  ///   - gradientClip: Optional gradient clipping threshold.
  ///   - augmenter: Optional training-time data augmenter.
  public init(_ trainable: Trainable,
              learningRate: Tensor.Scalar,
              batchSize: Int,
              momentum: Tensor.Scalar = 0.9,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil,
              augmenter: Augmenter? = nil) {
    self.momentum = momentum
    
    trainable.compile()
    v = [TensorStorage].init(repeating: TensorStorage.create(count: 0), count: trainable.layers.count)
    vb = [TensorStorage].init(repeating: TensorStorage.create(count: 0), count: trainable.layers.count)
    
    super.init(trainable: trainable,
               learningRate: learningRate,
               batchSize: batchSize,
               weightClip: weightClip,
               gradientClip: gradientClip,
               augmenter: augmenter)
  }
  
  /// Applies one SGD update using the currently accumulated gradients.
  public override func step() {
    var gradients = gradientAccumulator.accumulate()

    if let clip = gradientClip {
      gradients = gradients.gradientL2NormClip(clip)
    }
    
    for i in 0..<trainable.layers.count {
      let layer = trainable.layers[i]
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
      
      var sgdGradient = (gradient, biasGradient)
      
      if layer.trainable, layer.usesOptimizer {
        sgdGradient = run(gradient: gradient, biasGradient: biasGradient, index: i)
      }
      
      layer.apply(gradients: sgdGradient, learningRate: learningRate)
      
      weightClip(layer: layer)
    }
  }
  
  private func run(gradient: Tensor, biasGradient: Tensor, index: Int) -> Optimizer.Gradient {
    let i = index

    if v[i].isEmpty {
      v[i] = TensorStorage.create(count: gradient.storage.count)
    }
    
    apply(to: &v[i], gradient: gradient.storage)

    if vb[i].isEmpty {
      vb[i] = TensorStorage.create(count: biasGradient.storage.count)
    }
          
    apply(to: &vb[i], gradient: biasGradient.storage)

    return (Tensor(storage: v[i], size: gradient.size),
            Tensor(storage: vb[i], size: biasGradient.size))
  }

  private func apply(to: inout TensorStorage, gradient: TensorStorage) {
    for i in 0..<gradient.count {
      to[i] = momentum * to[i] + learningRate * gradient[i]
    }
  }
  
  /// Clears internal velocity buffers and resets inherited optimizer state.
  public override func reset() {
    v.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }
}

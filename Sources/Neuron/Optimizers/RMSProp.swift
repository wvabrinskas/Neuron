//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/8/22.
//

import Foundation
import NumSwift

public class RMSProp: BaseOptimizer {
  private var b: Tensor.Scalar = 0.9
  private var v: [Tensor.Value] = []
  private var vb: [Tensor.Value] = []
  private var eps: Tensor.Scalar = .stabilityFactor

  /// Creates an RMSProp optimizer.
  ///
  /// - Parameters:
  ///   - trainable: Model whose parameters will be optimized.
  ///   - device: Execution device for forward/backward math.
  ///   - learningRate: Base learning rate.
  ///   - batchSize: Number of samples per optimization step.
  ///   - b: Exponential decay for moving squared-gradient average.
  ///   - eps: Numerical stability epsilon.
  ///   - weightClip: Optional weight clipping threshold.
  ///   - gradientClip: Optional gradient clipping threshold.
  ///   - augmenter: Optional training-time data augmenter.
  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              batchSize: Int,
              b: Tensor.Scalar = 0.9,
              eps: Tensor.Scalar = .stabilityFactor,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil,
              augmenter: Augmenter? = nil) {
    self.eps = eps
    self.b = b

    v = [Tensor.Value].init(repeating: Tensor.Value(), count: trainable.layers.count)
    vb = [Tensor.Value].init(repeating: Tensor.Value(), count: trainable.layers.count)
    
    trainable.compile()
    
    super.init(trainable: trainable,
               learningRate: learningRate,
               batchSize: batchSize,
               weightClip: weightClip,
               gradientClip: gradientClip,
               augmenter: augmenter)
  }
  
  /// Applies one RMSProp optimization step.
  public override func step() {
    var gradients = gradientAccumulator.accumulate()
    
    if let clip = gradientClip {
      gradients = gradients.gradientL2NormClip(clip)
    }
    
    for i in 0..<trainable.layers.count {
      let layer = trainable.layers[i]
      
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
  
      var adamGradient = (gradient, biasGradient)
      
      if layer.trainable, layer.usesOptimizer {
        adamGradient = run(gradient: gradient, biasGradient: biasGradient, index: i)
      }
      
      layer.apply(gradients: adamGradient, learningRate: learningRate)
      
      weightClip(layer: layer)
    }
  }
  
  /// Clears RMSProp running-stat buffers and inherited optimizer state.
  public override func reset() {
    v.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }

  private func run(gradient: Tensor, biasGradient: Tensor, index: Int) -> Optimizer.Gradient {
    let i = index
    
    if vb[i].isEmpty || v[i].isEmpty {
      v[i] = Tensor.Value(repeating: 0, count: gradient.storage.count)
      vb[i] = Tensor.Value(repeating: 0, count: biasGradient.storage.count)
    }
    
    let result = apply(to: &v[i], gradient: gradient.storage)
    let biasResult = apply(to: &vb[i], gradient: biasGradient.storage)
    
    return (Tensor(result, size: gradient.size), Tensor(biasResult, size: biasGradient.size))
  }

  private func apply(to: inout Tensor.Value, gradient: Tensor.Value) -> Tensor.Value {
    let count = gradient.count
    let oneMinusB = 1 - b
    var result = Tensor.Value(repeating: 0, count: count)

    for i in 0..<count {
      let g = gradient[i]
      to[i] = b * to[i] + oneMinusB * (g * g)
      
      let delta = (learningRate / (sqrt(to[i] + eps))) * g
      result[i] = delta
    }

    return result
  }
}

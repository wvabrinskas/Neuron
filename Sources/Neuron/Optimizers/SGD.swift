//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

public class SGD: BaseOptimizer {
  private let momentum: Tensor.Scalar
  private var v: [Tensor.Value] = []
  private var vb: [Tensor.Value] = []

  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              batchSize: Int,
              momentum: Tensor.Scalar = 0.9,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil) {
    self.momentum = momentum
    
    trainable.compile()
    v = [Tensor.Value].init(repeating: Tensor.Value(), count: trainable.layers.count)
    vb = [Tensor.Value].init(repeating: Tensor.Value(), count: trainable.layers.count)
    
    super.init(trainable: trainable,
               learningRate: learningRate,
               batchSize: batchSize,
               weightClip: weightClip,
               gradientClip: gradientClip)
  }
  
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
      v[i] = Tensor.Value(repeating: 0, count: gradient.storage.count)
    }
    
    apply(to: &v[i], gradient: gradient.storage)

    if vb[i].isEmpty {
      vb[i] = Tensor.Value(repeating: 0, count: biasGradient.storage.count)
    }
          
    apply(to: &vb[i], gradient: biasGradient.storage)

    return (Tensor(v[i], size: gradient.size),
            Tensor(vb[i], size: biasGradient.size))
  }

  private func apply(to: inout Tensor.Value, gradient: Tensor.Value) {
    for i in 0..<gradient.count {
      to[i] = momentum * to[i] + learningRate * gradient[i]
    }
  }
  
  public override func reset() {
    v.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }
}

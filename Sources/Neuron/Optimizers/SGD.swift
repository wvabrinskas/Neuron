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
  private var v: [Tensor.Data] = []
  private var vb: [Tensor.Data] = []

  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              batchSize: Int,
              momentum: Tensor.Scalar = 0.9,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil) {
    self.momentum = momentum
    
    trainable.compile()
    v = [[[[Tensor.Scalar]]]].init(repeating: [], count: trainable.layers.count)
    vb = [Tensor.Data].init(repeating: [], count: trainable.layers.count)
    
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
    let shape = gradient.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    let depth = shape[safe: 2] ?? 0
    let i = index
    
    let biasShape = biasGradient.shape
    let biasRows = biasShape[safe: 1] ?? 0
    let biasColumns = biasShape[safe: 0] ?? 0
    let biasDepth = biasShape[safe: 2] ?? 0

    if v[i].isEmpty {
      v[i] = NumSwift.zerosLike((rows, columns, depth))
    }
    
    let gradientValue = gradient.value
    apply(to: &v[i], gradient: gradientValue)

    if vb[i].isEmpty {
      vb[i] = NumSwift.zerosLike((biasRows, biasColumns, biasDepth))
    }
          
    let biasValue = biasGradient.value
    apply(to: &vb[i], gradient: biasValue)

    return (Tensor(v[i]), Tensor(vb[i]))
  }

  private func apply(to: inout Tensor.Data, gradient: Tensor.Data) {
    for d in 0..<gradient.count {
      let depth = gradient[d]
      for r in 0..<depth.count {
        let row = depth[r]
        for c in 0..<row.count {
          to[d][r][c] = momentum * to[d][r][c] + learningRate * gradient[d][r][c]
        }
      }
    }
  }
  
  public override func reset() {
    v.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }
}

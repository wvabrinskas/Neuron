//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

public class SGD: BaseOptimizer {
  private let momentum: Float
  private var v: [Tensor.Data] = []
  private var vb: [[Tensor.Scalar]] = []

  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Float,
              momentum: Float = 0.9,
              l2Normalize: Bool = false) {
    self.momentum = momentum
    
    trainable.compile()
    v = [[[[Tensor.Scalar]]]].init(repeating: [], count: trainable.layers.count)
    vb = [[Tensor.Scalar]].init(repeating: [], count: trainable.layers.count)
    
    super.init(trainable: trainable, learningRate: learningRate, l2Normalize: l2Normalize)
  }
  
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
  
  public override func reset() {
    v.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }
}

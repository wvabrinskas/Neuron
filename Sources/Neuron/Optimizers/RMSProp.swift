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
  private var v: [Tensor.Data] = []
  private var vb: [[Tensor.Scalar]] = []
  private var eps: Tensor.Scalar = .stabilityFactor

  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              b: Tensor.Scalar = 0.9,
              eps: Tensor.Scalar = .stabilityFactor,
              threadWorkers: Int = 16) {
    self.eps = eps
    self.b = b

    v = [[[[Tensor.Scalar]]]].init(repeating: [], count: trainable.layers.count)
    vb = [[Tensor.Scalar]].init(repeating: [], count: trainable.layers.count)
    
    trainable.compile()
    
    super.init(trainable: trainable,
               learningRate: learningRate,
               l2Normalize: false,
               workers: threadWorkers)
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

      var adamGradient = (gradient, biasGradient)
      
      if layer.trainable, layer.usesOptimizer {
        adamGradient = run(gradient: gradient, biasGradient: biasGradient, index: i)
      }
      
      layer.apply(gradients: adamGradient, learningRate: learningRate)
      
      clip(layer: layer)
    }
  }
  
  public override func reset() {
    v.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }

  private func run(gradient: Tensor, biasGradient: Tensor, index: Int) -> Optimizer.Gradient {
    let shape = gradient.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    let depth = shape[safe: 2] ?? 0
    let i = index
    
    let flatBias = biasGradient.value.flatten()

    var result: [[[Tensor.Scalar]]] = NumSwift.zerosLike((rows, columns, depth))
    var biases: [Tensor.Scalar] = .init(repeating: 0, count: flatBias.count)
    
    if vb[i].isEmpty || v[i].isEmpty {
      v[i] = NumSwift.zerosLike((rows, columns, depth))
      vb[i] = [Tensor.Scalar].init(repeating: 0, count: flatBias.count)
    }
    
    let gradientValue = gradient.value
    
    for d in 0..<gradientValue.count {
      let depthGradient = gradientValue[d]
      for r in 0..<depthGradient.count {
        let rowGradient = depthGradient[r]
        for c in 0..<rowGradient.count {
          v[i][d][r][c] = b * v[i][d][r][c] + (1 - b) * Tensor.Scalar.pow(gradientValue[d][r][c], 2)
          
          let vdw = v[i][d][r][c]
          let alpha = learningRate
          let dw = gradientValue[d][r][c]
          
          let delta = (alpha / (sqrt(vdw + eps))) * dw
          
          result[d][r][c] = delta
        }
      }
    }
      
    
    for d in 0..<flatBias.count {
      // bias gradients are performed at a depth level
      let biasGradient = flatBias[d]
      vb[i][d] = b * vb[i][d] + (1 - b) * Tensor.Scalar.pow(biasGradient, 2)
      let vdb = vb[i][d]
      let alpha = learningRate
      let db = biasGradient
      
      let deltaB = (alpha / (sqrt(vdb + eps))) * db
      biases[d] = deltaB
    }
    
    return (Tensor(result), Tensor(biases))
  }
}

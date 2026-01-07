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
  private var vb: [Tensor.Data] = []
  private var eps: Tensor.Scalar = .stabilityFactor

  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              batchSize: Int,
              b: Tensor.Scalar = 0.9,
              eps: Tensor.Scalar = .stabilityFactor,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil) {
    self.eps = eps
    self.b = b

    v = [[[[Tensor.Scalar]]]].init(repeating: [], count: trainable.layers.count)
    vb = [[[[Tensor.Scalar]]]].init(repeating: [], count: trainable.layers.count)
    
    trainable.compile()
    
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
  
      var adamGradient = (gradient, biasGradient)
      
      if layer.trainable, layer.usesOptimizer {
        adamGradient = run(gradient: gradient, biasGradient: biasGradient, index: i)
      }
      
      layer.apply(gradients: adamGradient, learningRate: learningRate)
      
      weightClip(layer: layer)
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
    
    if vb[i].isEmpty || v[i].isEmpty {
      v[i] = NumSwift.zerosLike((rows, columns, depth))
      vb[i] = NumSwift.zerosLike((rows, columns, depth))
    }
    
    let gradientValue = gradient.value
    let result = apply(to: &v[i], gradient: gradientValue)
    
    let biasGradientValue = biasGradient.value
    let biasResult = apply(to: &vb[i], gradient: biasGradientValue)
    
    return (Tensor(result), Tensor(biasResult))
  }

  private func apply(to: inout Tensor.Data, gradient: Tensor.Data) -> [[[Tensor.Scalar]]] {
    var result: [[[Tensor.Scalar]]] = []
    
    for d in 0..<gradient.count {
      var row: [[Tensor.Scalar]] = []

      let depthGradient = gradient[d]
      for r in 0..<depthGradient.count {

        var column: [Tensor.Scalar] = []

        let rowGradient = depthGradient[r]
        for c in 0..<rowGradient.count {
          to[d][r][c] = b * to[d][r][c] + (1 - b) * Tensor.Scalar.pow(gradient[d][r][c], 2)
          
          let vdw = to[d][r][c]
          let alpha = learningRate
          let dw = gradient[d][r][c]
          
          let delta = (alpha / (sqrt(vdw + eps))) * dw
          
          column.append(delta)
        }
        row.append(column)
      }
      result.append(row)
    }

    return result
  }
}

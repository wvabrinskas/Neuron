//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// Convienence class for Adam with weight decay as default to `0.004`
public final class AdamW<N: TensorNumeric>: Adam<N> {
  public init(_ trainable: BaseTrainable<N>,
              device: Device = CPU(),
              learningRate: Tensor<N>.Scalar,
              b1: Tensor<N>.Scalar = 0.9,
              b2: Tensor<N>.Scalar = 0.999,
              eps: Tensor<N>.Scalar = .stabilityFactor,
              weightDecayValue: Tensor<N>.Scalar = 0.004) {
    super.init(trainable,
               device: device,
               learningRate: learningRate,
               b1: b1,
               b2: b2,
               eps: eps,
               weightDecay: .decay(weightDecayValue))
  }
}

public class Adam<N: TensorNumeric>: BaseOptimizer<N> {
  public enum WeightDecay {
    case none
    case decay(Tensor<N>.Scalar)
  }
  
  public override var trainable: BaseTrainable<N> {
    didSet {
      build()
    }
  }
  
  private var b1: Tensor<N>.Scalar = 0.9
  private var b2: Tensor<N>.Scalar = 0.999
  private var eps: Tensor<N>.Scalar = .stabilityFactor
  
  private var m: [Tensor<N>.Data] = []
  private var v: [Tensor<N>.Data] = []
  private var vb: [[Tensor<N>.Scalar]] = []
  private var mb: [[Tensor<N>.Scalar]] = []
  private var t: Tensor<N>.Scalar = 1
  private let weightDecay: WeightDecay
  
  public init(_ trainable: BaseTrainable<N>,
              device: Device = CPU(),
              learningRate: Tensor<N>.Scalar,
              b1: Tensor<N>.Scalar = 0.9,
              b2: Tensor<N>.Scalar = 0.999,
              eps: Tensor<N>.Scalar = .stabilityFactor,
              weightDecay: WeightDecay = .none) {
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.weightDecay = weightDecay
    super.init(trainable: trainable,
               learningRate: learningRate,
               l2Normalize: false,
               workers: 8) // limit to 8 because anything higher causes an extra second of runtime for some reason??
    build()
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
  
  private func build() {
    m = [Tensor<N>.Data].init(repeating: [], count: trainable.layers.count) // we want to support multiple weight structures right now this only supports one Tensor<N> for one m value, when layers could have multiple tensors representing weights
    v = [Tensor<N>.Data].init(repeating: [], count: trainable.layers.count)
    vb = [[Tensor<N>.Scalar]].init(repeating: [], count: trainable.layers.count)
    mb = [[Tensor<N>.Scalar]].init(repeating: [], count: trainable.layers.count)
    trainable.compile()
  }
  
  private func run(gradient: Tensor<N>, biasGradient: Tensor<N>, index: Int, weights: Tensor<N>) -> Optimizer.Gradient {
    let shape = gradient.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    let depth = shape[safe: 2] ?? 0
    let i = index
    
    let flatBias = biasGradient.value.flatten()

    var result: [[[Tensor<N>.Scalar]]] = []
    var biases: [Tensor<N>.Scalar] = .init(repeating: 0, count: flatBias.count)

    if m[i].isEmpty || v[i].isEmpty {
      m[i] = NumSwift.zerosLike((rows, columns, depth))
      v[i] = NumSwift.zerosLike((rows, columns, depth))
      mb[i] = [Tensor<N>.Scalar].init(repeating: 0, count: flatBias.count)
      vb[i] = [Tensor<N>.Scalar].init(repeating: 0, count: flatBias.count)
    }
    
    let gradientValue = gradient.value
        
    for d in 0..<gradientValue.count {
      let depthGradient = gradientValue[d]
      var row: [[Tensor<N>.Scalar]] = []
      for r in 0..<depthGradient.count {
        let rowGradient = depthGradient[r]
        var column: [Tensor<N>.Scalar] = []
        for c in 0..<rowGradient.count {
          m[i][d][r][c] = b1 * m[i][d][r][c] + (1 - b1) * gradientValue[d][r][c]
          v[i][d][r][c] = b2 * v[i][d][r][c] + (1 - b2) * Tensor<N>.Scalar.pow(gradientValue[d][r][c], 2)
          
          let mHat = m[i][d][r][c] / (1 - Tensor<N>.Scalar.pow(b1, Tensor<N>.Scalar(t)))
          let vHat = v[i][d][r][c] / (1 - Tensor<N>.Scalar.pow(b2, Tensor<N>.Scalar(t)))
          
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
      vb[i][d] = b2 * vb[i][d] + (1 - b2) * Tensor<N>.Scalar.pow(gradientSum, 2)
      
      let mHat = mb[i][d] / (1 - Tensor<N>.Scalar.pow(b1, Tensor<N>.Scalar(t)))
      let vHat = vb[i][d] / (1 - Tensor<N>.Scalar.pow(b2, Tensor<N>.Scalar(t)))
      
      let deltaB = learningRate * (mHat / (sqrt(vHat + eps)))
      
      biases[d] = deltaB
    }
        
    return (Tensor<N>(result), Tensor<N>(biases))
  }
  
  public override func reset() {
    t = 1
    m.removeAll(keepingCapacity: true)
    v.removeAll(keepingCapacity: true)
    mb.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
    super.reset()
  }
}

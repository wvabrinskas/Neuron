//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// Convienence class for Adam with weight decay as default to `0.004`
public final class AdamW: Adam {
  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              batchSize: Int,
              b1: Tensor.Scalar = 0.9,
              b2: Tensor.Scalar = 0.999,
              eps: Tensor.Scalar = .stabilityFactor,
              weightDecayValue: Tensor.Scalar = 0.004,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil) {
    super.init(trainable,
               device: device,
               learningRate: learningRate,
               batchSize: batchSize,
               b1: b1,
               b2: b2,
               eps: eps,
               weightDecay: .decay(weightDecayValue),
               weightClip: weightClip,
               gradientClip: gradientClip)
  }
}

public class Adam: BaseOptimizer {
  public enum WeightDecay {
    case none
    case decay(Tensor.Scalar)
  }
  
  public override var trainable: Trainable {
    didSet {
      build()
    }
  }
  
  private var b1: Tensor.Scalar = 0.9
  private var b2: Tensor.Scalar = 0.999
  private var eps: Tensor.Scalar = .stabilityFactor
  
  private var m: [Tensor.Data] = []
  private var v: [Tensor.Data] = []
  private var vb: [Tensor.Data] = []
  private var mb: [Tensor.Data] = []
  private var t: Tensor.Scalar = 1
  private let weightDecay: WeightDecay
  
  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Tensor.Scalar,
              batchSize: Int,
              b1: Tensor.Scalar = 0.9,
              b2: Tensor.Scalar = 0.999,
              eps: Tensor.Scalar = .stabilityFactor,
              weightDecay: WeightDecay = .none,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil) {
    self.b1 = b1
    self.b2 = b2
    self.eps = eps
    self.weightDecay = weightDecay
    super.init(trainable: trainable,
               learningRate: learningRate,
               batchSize: batchSize,
               weightClip: weightClip,
               gradientClip: gradientClip)
    build()
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
      
      var adamGradient: Gradient = (gradient, biasGradient)
      
      // only apply optimizer gradient if the layer is trainable by the optimizer
      if layer.trainable, layer.usesOptimizer {
        adamGradient = run(gradient: gradient, biasGradient: biasGradient, index: i, weights: layer.weights)
      }
            
      layer.apply(gradients: adamGradient, learningRate: learningRate)
      
      weightClip(layer: layer)
    }
    
    t += 1
    
    super.step()
  }
  
  private func build() {
    m = [Tensor.Data].init(repeating: [], count: trainable.layers.count) // we want to support multiple weight structures right now this only supports one Tensor for one m value, when layers could have multiple tensors representing weights
    v = [Tensor.Data].init(repeating: [], count: trainable.layers.count)
    vb = [Tensor.Data].init(repeating: [], count: trainable.layers.count)
    mb = [Tensor.Data].init(repeating: [], count: trainable.layers.count)
    trainable.compile()
  }
  
  private func run(gradient: Tensor, biasGradient: Tensor, index: Int, weights: Tensor) -> Optimizer.Gradient {
    let shape = gradient.shape
    let rows = shape[safe: 1] ?? 0
    let columns = shape[safe: 0] ?? 0
    let depth = shape[safe: 2] ?? 0
    let i = index

    if m[i].isEmpty || v[i].isEmpty {
      m[i] = NumSwift.zerosLike((rows, columns, depth))
      v[i] = NumSwift.zerosLike((rows, columns, depth))
    }
    
    let gradientValue = gradient.value
    let result = apply(m: &m[i],
                                    v: &v[i],
                                    gradient: gradientValue,
                                    decay: true,
                                    weights: weights)

    let biasShape = biasGradient.shape
    let biasRows = biasShape[safe: 1] ?? 0
    let biasColumns = biasShape[safe: 0] ?? 0
    let biasDepth = biasShape[safe: 2] ?? 0

    if mb[i].isEmpty || vb[i].isEmpty {
      mb[i] = NumSwift.zerosLike((biasRows, biasColumns, biasDepth))
      vb[i] = NumSwift.zerosLike((biasRows, biasColumns, biasDepth))
    }

    // Get the bias gradient value
    let biasGradientValue = biasGradient.value
    let biases = apply(m: &mb[i],
                                    v: &vb[i],
                                    gradient: biasGradientValue,
                                    weights: weights)
    
    return (Tensor(result), Tensor(biases))
  }

  private func apply(m: inout Tensor.Data,
                     v: inout Tensor.Data,
                     gradient: Tensor.Data,
                     decay: Bool = false,
                     weights: Tensor) -> Tensor.Data {

    var result: [[[Tensor.Scalar]]] = []

    for d in 0..<gradient.count {
      let depthValue = gradient[d]
      var row: [[Tensor.Scalar]] = []
      for r in 0..<depthValue.count {
        let rowValue = depthValue[r]
        var column: [Tensor.Scalar] = []
        for c in 0..<rowValue.count {
          m[d][r][c] = b1 * m[d][r][c] + (1 - b1) * gradient[d][r][c]
          v[d][r][c] = b2 * v[d][r][c] + (1 - b2) * Tensor.Scalar.pow(gradient[d][r][c], 2)

          let mHat = m[d][r][c] / (1 - Tensor.Scalar.pow(b1, Tensor.Scalar(t)))
          let vHat = v[d][r][c] / (1 - Tensor.Scalar.pow(b2, Tensor.Scalar(t)))

          var delta = learningRate * (mHat / (sqrt(vHat + eps)))
          
          if decay, case .decay(let decay) = weightDecay {
            let decayLR = learningRate * decay * weights.value[safe: d, [[]]][safe: r, []][safe: c, 1]
            delta -= decayLR
          }

          column.append(delta)
        }
        row.append(column)
      }
      result.append(row)
    }

    return result
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

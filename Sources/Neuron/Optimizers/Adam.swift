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
  
  private var m: [ContiguousArray<Tensor.Scalar>] = []
  private var v: [ContiguousArray<Tensor.Scalar>] = []
  private var vb: [ContiguousArray<Tensor.Scalar>] = []
  private var mb: [ContiguousArray<Tensor.Scalar>] = []
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
    m = [ContiguousArray<Tensor.Scalar>].init(repeating: ContiguousArray(), count: trainable.layers.count)
    v = [ContiguousArray<Tensor.Scalar>].init(repeating: ContiguousArray(), count: trainable.layers.count)
    vb = [ContiguousArray<Tensor.Scalar>].init(repeating: ContiguousArray(), count: trainable.layers.count)
    mb = [ContiguousArray<Tensor.Scalar>].init(repeating: ContiguousArray(), count: trainable.layers.count)
    trainable.compile()
  }
  
  private func run(gradient: Tensor, biasGradient: Tensor, index: Int, weights: Tensor) -> Optimizer.Gradient {
    let i = index
    let gradCount = gradient.storage.count

    if m[i].isEmpty || v[i].isEmpty {
      m[i] = ContiguousArray<Tensor.Scalar>(repeating: 0, count: gradCount)
      v[i] = ContiguousArray<Tensor.Scalar>(repeating: 0, count: gradCount)
    }
    
    let result = apply(m: &m[i],
                       v: &v[i],
                       gradient: gradient.storage,
                       decay: true,
                       weights: weights.storage,
                       size: gradient._size)

    let biasCount = biasGradient.storage.count

    if mb[i].isEmpty || vb[i].isEmpty {
      mb[i] = ContiguousArray<Tensor.Scalar>(repeating: 0, count: biasCount)
      vb[i] = ContiguousArray<Tensor.Scalar>(repeating: 0, count: biasCount)
    }

    let biases = apply(m: &mb[i],
                       v: &vb[i],
                       gradient: biasGradient.storage,
                       weights: weights.storage,
                       size: biasGradient._size)
    
    return (Tensor(result, size: gradient._size), Tensor(biases, size: biasGradient._size))
  }

  private func apply(m: inout ContiguousArray<Tensor.Scalar>,
                     v: inout ContiguousArray<Tensor.Scalar>,
                     gradient: ContiguousArray<Tensor.Scalar>,
                     decay: Bool = false,
                     weights: ContiguousArray<Tensor.Scalar>,
                     size: TensorSize) -> ContiguousArray<Tensor.Scalar> {

    // Hoist loop-invariant computations
    let oneMinusB1 = 1 - b1
    let oneMinusB2 = 1 - b2
    let mCorrectionFactor = 1 / (1 - Tensor.Scalar.pow(b1, Tensor.Scalar(t)))
    let vCorrectionFactor = 1 / (1 - Tensor.Scalar.pow(b2, Tensor.Scalar(t)))
    let shouldDecay: Tensor.Scalar? = decay ? {
      if case .decay(let d) = weightDecay { return d }
      return nil
    }() : nil

    let count = gradient.count
    var result = ContiguousArray<Tensor.Scalar>(repeating: 0, count: count)

    for i in 0..<count {
      let g = gradient[i]
      m[i] = b1 * m[i] + oneMinusB1 * g
      v[i] = b2 * v[i] + oneMinusB2 * (g * g)

      let mHat = m[i] * mCorrectionFactor
      let vHat = v[i] * vCorrectionFactor

      var delta = learningRate * (mHat / (sqrt(vHat + eps)))

      if let decayValue = shouldDecay, i < weights.count {
        let decayLR = learningRate * decayValue * weights[i]
        delta -= decayLR
      }

      result[i] = delta
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

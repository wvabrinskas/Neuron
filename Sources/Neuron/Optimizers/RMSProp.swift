//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/8/22.
//

import Foundation
import NumSwift

public class RMSProp: Optimizer {
  public let gradientAccumulator: GradientAccumulator = .init()
  public var clip: Float?
  public var trainable: Trainable
  public var learningRate: Float
  public var isTraining: Bool = true {
    didSet {
      trainable.isTraining = isTraining
    }
  }
  public var device: Device = CPU() {
    didSet {
      switch device.type {
      case .cpu:
        trainable.device = CPU()
      case .gpu:
        trainable.device = GPU()
      }
    }
  }
  public var l2Normalize: Bool
  public var workers: Int = 8
  public var metricsReporter: MetricsReporter?

  private var b: Tensor.Scalar = 0.9
  private var v: [Tensor.Data] = []
  private var vb: [[Tensor.Scalar]] = []
  private var eps: Float = 1e-8

  public init(_ trainable: Trainable,
              device: Device = CPU(),
              learningRate: Float,
              b: Float = 0.9,
              eps: Float = 1e-8,
              l2Normalize: Bool = false) {
    self.trainable = trainable
    self.eps = eps
    self.b = b
    self.learningRate = learningRate
    self.device = device
    self.l2Normalize = l2Normalize
    v = [[[[Tensor.Scalar]]]].init(repeating: [], count: trainable.layers.count)
    vb = [[Tensor.Scalar]].init(repeating: [], count: trainable.layers.count)

    trainable.compile()
  }
  
  public func step() {
    let gradients = gradientAccumulator.accumulate()
    
    for i in 0..<trainable.layers.count {
      let layer = trainable.layers[i]
      
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
      
      if l2Normalize {
        gradient.l2Normalize()
      }

      var adamGradient = (gradient, biasGradient)
      
      if layer.trainable {
        adamGradient = run(gradient: gradient, biasGradient: biasGradient, index: i)
      }
      
      layer.apply(gradients: adamGradient, learningRate: learningRate)
      
      clip(layer: layer)
    }
  }
  
  public func reset() {
    v.removeAll(keepingCapacity: true)
    vb.removeAll(keepingCapacity: true)
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
          v[i][d][r][c] = b * v[i][d][r][c] + (1 - b) * pow(gradientValue[d][r][c], 2)
          
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
      vb[i][d] = b * vb[i][d] + (1 - b) * pow(biasGradient, 2)
      let vdb = vb[i][d]
      let alpha = learningRate
      let db = biasGradient
      
      let deltaB = (alpha / (sqrt(vdb + eps))) * db
      biases[d] = deltaB
    }
    
    return (Tensor(result), Tensor(biases))
  }
}

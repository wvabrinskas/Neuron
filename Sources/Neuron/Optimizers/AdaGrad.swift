//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import NumSwift

/// An optimizer that implements the AdaGrad (Adaptive Gradient) algorithm.
///
/// Accumulates squared gradients per parameter and scales the learning rate inversely
/// by the square root of the accumulated sums, giving infrequent parameters larger updates.
public class AdaGrad: BaseOptimizer {
  private var g: [Tensor] = []
  private var gb: [Tensor] = []

  /// Creates an AdaGrad optimizer.
  ///
  /// - Parameters:
  ///   - trainable: Model whose parameters will be optimized.
  ///   - learningRate: Base learning rate scaled adaptively per parameter.
  ///   - batchSize: Number of samples per optimization step.
  ///   - weightClip: Optional weight clipping threshold.
  ///   - gradientClip: Optional gradient clipping threshold.
  ///   - augmenter: Optional training-time data augmenter.
  public init(_ trainable: Trainable,
              learningRate: Tensor.Scalar,
              batchSize: Int,
              weightClip: Tensor.Scalar? = nil,
              gradientClip: Tensor.Scalar? = nil,
              augmenter: Augmenter? = nil) {
    
    trainable.compile()
    
    for layer in trainable.layers {
      g.append(Tensor.fillWith(value: 0, size: layer.weights.size))
      gb.append(Tensor.fillWith(value: 0, size: layer.biases.size))
    }

    super.init(trainable: trainable,
               learningRate: learningRate,
               batchSize: batchSize,
               weightClip: weightClip,
               gradientClip: gradientClip,
               augmenter: augmenter)
  }
  
  /// Applies one AdaGrad optimization step using the accumulated gradients.
  public override func step() {
    var gradients = gradientAccumulator.accumulate()

    if let clip = gradientClip {
      gradients = gradients.gradientL2NormClip(clip)
    } else if metricsReporter?.metricsToGather.contains(.globalGradientNorm) == true {
      gradients.calculateL2Norm(metrics: metricsReporter)
    }
    
    for i in 0..<trainable.layers.count {
      let layer = trainable.layers[i]
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
      
      let gLocal = gradient * gradient
      let gbLocal = biasGradient * biasGradient
      
      g[i] = g[i].copy() + gLocal
      gb[i] = gb[i].copy() + gbLocal
      
      var adaGradient = (gradient, biasGradient)
      
      if layer.trainable, layer.usesOptimizer {
        adaGradient = run(gradient: gradient, biasGradient: biasGradient, index: i)
      }
      
      layer.apply(gradients: adaGradient, learningRate: learningRate)
      
      weightClip(layer: layer)
    }
  }
  
  private func run(gradient: Tensor, biasGradient: Tensor, index: Int) -> Optimizer.Gradient {
    let gLocal = g[index]
    let gbLocal = gb[index]
    
    let weightGradient = (learningRate / gLocal.sqrt(adding: .stabilityFactor)) * gradient
    let biasGradient = (learningRate / gbLocal.sqrt(adding: .stabilityFactor)) * biasGradient

    return (weights: weightGradient, biases: biasGradient)
  }
  
  /// Clears internal velocity buffers and resets inherited optimizer state.
  public override func reset() {
    for layer in trainable.layers {
      g.append(Tensor.fillWith(value: 0, size: layer.weights.size))
      gb.append(Tensor.fillWith(value: 0, size: layer.biases.size))
    }
    super.reset()
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift
import Numerics

/// A learning-rate schedule that anneals the learning rate following a cosine curve.
///
/// The learning rate is decreased from `maxLR` to `minLR` over the specified number of epochs
/// using a cosine annealing strategy.
public final class CosineAnnealingDecay: BaseDecayFunction {
  private let minLR: Tensor.Scalar
  private let maxLR: Tensor.Scalar
  
  
  /// Creates a cosine annealing learning-rate schedule.
  ///
  /// - Parameters:
  ///   - learningRate: Maximum (initial) learning rate.
  ///   - minLearningRate: Minimum learning rate at the end of each cycle.
  ///   - decaySteps: Number of steps in one annealing cycle.
  public init(learningRate: Tensor.Scalar,
              minLearningRate: Tensor.Scalar,
              decaySteps: Int) {
    self.minLR = minLearningRate
    self.maxLR = learningRate
    
    super.init(learningRate: learningRate,
               decayRate: 0,
               decaySteps: Tensor.Scalar(decaySteps))
  }
  
  /// Advances the schedule and computes the next decayed learning rate.
  public override func step() {
    let cosine = Tensor.Scalar.cos(Tensor.Scalar.pi * Tensor.Scalar(globalSteps) / Tensor.Scalar(decaySteps))
    let adjusted = minLR + 0.5 * (maxLR - minLR) * (1 + cosine)
    decayedLearningRate = adjusted
    super.step()
  }
}

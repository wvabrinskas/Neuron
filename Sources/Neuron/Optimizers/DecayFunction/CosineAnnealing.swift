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
  
  
  /// Creates an exponential learning-rate decay schedule.
  ///
  /// - Parameters:
  ///   - learningRate: Initial learning rate.
  ///   - decayRate: Multiplicative factor applied every `decaySteps`.
  ///   - decaySteps: Step interval used by the decay equation.
  ///   - staircase: When `true`, uses floor-stepped exponent updates.
  public init(learningRate: Tensor.Scalar,
              minLearningRate: Tensor.Scalar,
              decaySteps: Int) {
    self.minLR = minLearningRate
    self.maxLR = learningRate
    
    super.init(learningRate: learningRate,
               decayRate: 0,
               decaySteps: Tensor.Scalar(decaySteps),
               staircase: false)
  }
  
  /// Advances the schedule and computes the next decayed learning rate.
  public override func step() {
    let cosine = Tensor.Scalar.cos(Tensor.Scalar.pi * Tensor.Scalar(globalSteps) / Tensor.Scalar(decaySteps))
    let adjusted = minLR + 0.5 * (maxLR - minLR) * (1 + cosine)
    
    decayedLearningRate = adjusted
    
    super.step()
  }
}

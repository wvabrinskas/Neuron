//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift
import Numerics

/// A learning-rate scheduler that applies exponential decay over training steps.
public final class ExponentialDecay: BaseDecayFunction {
  private var staircase: Bool
  /// Creates an exponential learning-rate decay schedule.
  ///
  /// - Parameters:
  ///   - learningRate: Initial learning rate.
  ///   - decayRate: Multiplicative factor applied every `decaySteps`.
  ///   - decaySteps: Step interval used by the decay equation.
  ///   - staircase: When `true`, uses floor-stepped exponent updates.
  public init(learningRate: Tensor.Scalar,
                       decayRate: Tensor.Scalar = 0.96,
                       decaySteps: Tensor.Scalar = 1000,
                       staircase: Bool = false) {
    self.staircase = staircase
    
    super.init(learningRate: learningRate,
               decayRate: decayRate,
               decaySteps: decaySteps)
  }
  
  /// Advances the schedule and computes the next decayed learning rate.
  public override func step() {
    super.step()
    
    let exp: Tensor.Scalar
    if staircase {
      exp = Tensor.Scalar(Int(globalSteps) / Int(decaySteps))
    } else {
      exp = globalSteps / decaySteps
    }
    
    decayedLearningRate = originalLearningRate * Tensor.Scalar.pow(decayRate, exp)
  }
}

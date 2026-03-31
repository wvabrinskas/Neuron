//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift
import Numerics

public final class LinearDecay: BaseDecayFunction {
  public init(learningRate: Tensor.Scalar,
              decaySteps: Tensor.Scalar = 1000) {
    super.init(learningRate: learningRate,
               decayRate: 0,
               decaySteps: decaySteps)
  }
  
  /// Advances the schedule and computes the next decayed learning rate.
  public override func step() {
    decayedLearningRate = originalLearningRate * (1 - (globalSteps / decaySteps))
    super.step()
  }
}

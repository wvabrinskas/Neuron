//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift
import Numerics

/// A learning rate decay schedule that reduces the learning rate linearly over a fixed number of steps.
///
/// The decayed rate is computed as: `decayedLearningRate = originalLearningRate * (1 - globalSteps / decaySteps)`.
public final class LinearDecay: BaseDecayFunction {
  /// Creates a `LinearDecay` schedule.
  /// - Parameters:
  ///   - learningRate: The initial learning rate before any decay is applied.
  ///   - decaySteps: The total number of steps over which the learning rate decays to zero. Defaults to `1000`.
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

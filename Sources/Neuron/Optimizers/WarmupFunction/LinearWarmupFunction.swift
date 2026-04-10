//
//  LinearWarmupFunction.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation

/// A warmup schedule that linearly increases the learning rate from near-zero to the target value.
///
/// The warmed rate is computed as: `targetLearningRate * (globalSteps / warmupSteps)`.
public class LinearWarmupFunction: BaseWarmupFunction {
  /// Advances the warmup schedule by one step and updates `warmedLearningRate` using linear interpolation.
  public override func step() {
    warmedLearningRate = targetLearningRate * (Tensor.Scalar(globalSteps) / Tensor.Scalar(warmupSteps))
    super.step()
  }
}

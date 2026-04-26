//
//  LinearWarmupFunction.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation

/// A warmup schedule that increases the learning rate using a cosine curve.
///
/// The warmed rate is computed as: `targetLearningRate * 0.5 * (1 - cos(π * step / warmupSteps))`.
/// This produces a smooth, gradual increase that avoids the sharp ramp of linear warmup.
public class CosineWarmupFunction: BaseWarmupFunction {
  /// Advances the warmup schedule by one step and updates `warmedLearningRate` using the cosine formula.
  public override func step() {
    warmedLearningRate = targetLearningRate * 0.5 * (1 - Tensor.Scalar.cos(Tensor.Scalar.pi * Tensor.Scalar(globalSteps) / Tensor.Scalar(warmupSteps)))
    super.step()
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift

public enum DecayStepType {
  case batch, epoch(Int)
}

public protocol DecayFunction {
  var decayedLearningRate: Tensor.Scalar { get }
  /// Resets decay state back to its initial learning-rate value.
  func reset()
  /// Advances decay state by one optimization step.
  func step(type: DecayStepType)
}

open class BaseDecayFunction: DecayFunction {
  public var decayedLearningRate: Tensor.Scalar
  
  let originalLearningRate: Tensor.Scalar
  let decayRate: Tensor.Scalar
  let decaySteps: Tensor.Scalar
  var globalSteps: Tensor.Scalar = 0
  let staircase: Bool
  
  /// Creates a base learning-rate decay schedule.
  ///
  /// - Parameters:
  ///   - learningRate: Initial learning rate before decay.
  ///   - decayRate: Multiplicative decay factor.
  ///   - decaySteps: Number of steps per decay unit.
  ///   - staircase: Whether to apply discrete staircase decay.
  public init(learningRate: Tensor.Scalar,
              decayRate: Tensor.Scalar,
              decaySteps: Tensor.Scalar,
              staircase: Bool) {
    self.originalLearningRate = learningRate
    self.decayedLearningRate = learningRate
    self.decayRate = decayRate
    self.decaySteps = decaySteps
    self.staircase = staircase
  }
  
  /// Restores the original learning rate and step counter.
  open func reset() {
    decayedLearningRate = originalLearningRate
    globalSteps = 0
  }
  
  /// Increments the global step counter.
  ///
  /// Subclasses should override to compute `decayedLearningRate` and then call
  /// `super.step()` to keep step accounting in sync.
  open func step(type: DecayStepType) {
    // override and apply function here
    globalSteps += 1
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift

/// A protocol defining a learning-rate decay schedule.
///
/// Types conforming to `DecayFunction` provide a decayed learning rate
/// and support resetting and stepping through the decay schedule.
public protocol DecayFunction {
  /// The current decayed learning rate produced by this schedule.
  var decayedLearningRate: Tensor.Scalar { get }
  /// Resets decay state back to its initial learning-rate value.
  func reset()
  /// Advances decay state by one optimization step.
  func step()
}

/// A base class that provides common state and bookkeeping for learning-rate decay schedules.
///
/// Subclasses should override `step()` to implement a specific decay formula,
/// update `decayedLearningRate`, and call `super.step()` to keep `globalSteps` in sync.
open class BaseDecayFunction: DecayFunction {
/// The current decayed learning rate value.
  public var decayedLearningRate: Tensor.Scalar
  
  let originalLearningRate: Tensor.Scalar
  let decayRate: Tensor.Scalar
  let decaySteps: Tensor.Scalar
  var globalSteps: Tensor.Scalar = 0
  
  /// Creates a base learning-rate decay schedule.
  ///
  /// - Parameters:
  ///   - learningRate: Initial learning rate before decay.
  ///   - decayRate: Multiplicative decay factor.
  ///   - decaySteps: Number of steps per decay unit.
  ///   - staircase: Whether to apply discrete staircase decay.
  public init(learningRate: Tensor.Scalar,
              decayRate: Tensor.Scalar,
              decaySteps: Tensor.Scalar) {
    self.originalLearningRate = learningRate
    self.decayedLearningRate = learningRate
    self.decayRate = decayRate
    self.decaySteps = decaySteps
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
  open func step() {
    // override and apply function here
    globalSteps += 1
  }
}

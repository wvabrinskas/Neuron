//
//  WarmupFunction.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation

/// Represents the current phase of a learning rate warmup schedule.
public enum WarmupState {
  case warming, complete
}

/// Defines the interface for learning rate warmup strategies.
///
/// Conforming types ramp the learning rate from a small initial value up to a
/// target value over a fixed number of warmup steps.
public protocol WarmupFunction {
  /// The current learning rate produced by this warmup schedule.
  var warmedLearningRate: Tensor.Scalar { get }
  /// The current phase of the warmup schedule — `.warming` or `.complete`.
  var warmupState: WarmupState { get }
  /// Resets the warmup schedule to its initial (zero) learning rate.
  func reset()
  /// Advances the warmup schedule by one step, updating `warmedLearningRate`.
  func step()
}

/// A base class providing common state for learning rate warmup schedules.
///
/// Subclasses should override `step()` to implement a specific warmup formula,
/// update `warmedLearningRate`, and call `super.step()` to advance `globalSteps`.
open class BaseWarmupFunction: WarmupFunction {
  /// The current warmed learning rate, updated on each call to `step()`.
  public var warmedLearningRate: Tensor.Scalar = Tensor.Scalar.stabilityFactor
  
  /// The current phase of the warmup schedule — `.warming` while below target, `.complete` once reached.
  public var warmupState: WarmupState {
    warmedLearningRate >= targetLearningRate ? .complete : .warming
  }
  
  let targetLearningRate: Tensor.Scalar
  let warmupSteps: Tensor.Scalar
  var globalSteps: Tensor.Scalar = 0
  
  /// Creates a `BaseWarmupFunction` with the specified target learning rate and number of warmup steps.
  /// - Parameters:
  ///   - targetLearningRate: The learning rate value to ramp up to by the end of warmup.
  ///   - warmupSteps: The total number of steps over which warmup is applied.
  public init(targetLearningRate: Tensor.Scalar,
              warmupSteps: Tensor.Scalar) {
    self.targetLearningRate = targetLearningRate
    self.warmupSteps = warmupSteps
  }
  
  /// Resets the warmup schedule to its initial state, zeroing the learning rate and step counter.
  open func reset() {
    warmedLearningRate = 0
    globalSteps = 0
  }

  /// Advances the warmup schedule by one step.
  ///
  /// Subclasses should override this method to update `warmedLearningRate` using their
  /// specific warmup formula, then call `super.step()` to increment `globalSteps`.
  open func step() {
    // override and apply function here
    globalSteps += 1
  }
}

//
//  LearningRateScheduling.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation


/// Defines the granularity at which a decay step is applied.
///
/// - `batch`: Decay is applied once per batch.
/// - `epoch`: Decay is applied once per epoch, with an associated integer epoch index.
public enum LearningRateScheduleStepType: Equatable {
  case batch, epoch
}

/// Defines the interface for a combined warmup + decay learning rate schedule.
///
/// Implementations first apply a warmup phase to ramp the learning rate up,
/// then hand off to a decay function once warmup is complete.
public protocol LearningRateScheduling {
  /// The current effective learning rate produced by the active schedule phase.
  var learningRate: Tensor.Scalar { get }
  /// The warmup schedule applied during the initial training phase.
  var warmup: WarmupFunction { get }
  /// The decay schedule applied once warmup is complete.
  var decay: DecayFunction { get }
  /// Advances the schedule by one step of the specified type.
  ///
  /// - Parameter type: Whether the step corresponds to a batch or an epoch boundary.
  func step(type: LearningRateScheduleStepType)
  /// Resets the scheduler, warmup, and decay to their initial states.
  func reset()
}

/// A learning rate scheduler that sequentially applies warmup followed by decay.
///
/// During warmup the learning rate is driven by the provided `WarmupFunction`.
/// Once warmup completes, the `DecayFunction` takes over for the remainder of training.
public final class SequentialLearningRateScheduler: LearningRateScheduling {
  /// The current effective learning rate, updated on every call to `step(type:)`.
  public var learningRate: Tensor.Scalar
  /// The warmup schedule used during the initial training phase.
  public let warmup: WarmupFunction
  /// The decay schedule applied after warmup completes.
  public let decay: DecayFunction
  
  private let type: LearningRateScheduleStepType
  
  private let metricsReporter: MetricsReporter?
  
  /// Creates a `SequentialLearningRateScheduler`.
  /// - Parameters:
  ///   - learningRate: The starting learning rate passed to the warmup function.
  ///   - warmup: The warmup schedule to apply first.
  ///   - decay: The decay schedule to apply after warmup completes.
  ///   - type: The step granularity (`.batch` or `.epoch`) that triggers an update.
  public init(learningRate: Tensor.Scalar,
              warmup: WarmupFunction,
              decay: DecayFunction,
              type: LearningRateScheduleStepType,
              metricsReporter: MetricsReporter? = nil) {
    self.learningRate = learningRate
    self.warmup = warmup
    self.decay = decay
    self.type = type
    self.metricsReporter = metricsReporter
    
    reset()
  }
  
  /// Advances the schedule by one step if the provided `type` matches the scheduler's configured step type.
  ///
  /// Delegates to the warmup function until warmup is complete, then delegates to the decay function.
  /// - Parameter type: The step type triggering this update (`.batch` or `.epoch`).
  public func step(type: LearningRateScheduleStepType) {
    guard type == self.type else { return }
    
    if warmup.warmupState != .complete {
      warmup.step()
      learningRate = warmup.warmedLearningRate
    } else {
      decay.step()
      learningRate = decay.decayedLearningRate
    }
    
    metricsReporter?.update(metric: .currentLearningRate, value: learningRate)
  }
  
  /// Resets the scheduler, warmup, and decay back to their initial states.
  public func reset() {
    warmup.reset()
    decay.reset()
    
    learningRate = warmup.warmedLearningRate
    
    metricsReporter?.update(metric: .currentLearningRate, value: learningRate)
  }

}


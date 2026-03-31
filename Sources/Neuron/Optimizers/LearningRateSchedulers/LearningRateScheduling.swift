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
  case batch, epoch(Int)
}

public protocol LearningRateScheduling {
  var learningRate: Tensor.Scalar { get }
  var warmup: WarmupFunction { get }
  var decay: DecayFunction { get }
  func step(type: LearningRateScheduleStepType)
  func reset()
}

public final class SequentialLearningRateScheduler: LearningRateScheduling {
  public var learningRate: Tensor.Scalar
  public let warmup: WarmupFunction
  public let decay: DecayFunction
  
  private let type: LearningRateScheduleStepType
  
  public init(learningRate: Tensor.Scalar,
              warmup: WarmupFunction,
              decay: DecayFunction,
              type: LearningRateScheduleStepType) {
    self.learningRate = learningRate
    self.warmup = warmup
    self.decay = decay
    self.type = type
  }
  
  public func step(type: LearningRateScheduleStepType) {
    guard type == self.type else { return }
    
    if warmup.warmupState != .complete {
      warmup.step()
      learningRate = warmup.warmedLearningRate
    } else {
      decay.step()
      learningRate = decay.decayedLearningRate
    }
  }
  
  public func reset() {
    warmup.reset()
    decay.reset()
    
    learningRate = warmup.warmedLearningRate
  }

}


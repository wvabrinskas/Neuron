//
//  WarmupFunction.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation

public enum WarmupState {
  case warming, complete
}

public protocol WarmupFunction {
  var warmedLearningRate: Tensor.Scalar { get }
  var warmupState: WarmupState { get }
  func reset()
  func step()
}

open class BaseWarmupFunction: WarmupFunction {
/// The current decayed learning rate value.
  public var warmedLearningRate: Tensor.Scalar = Tensor.Scalar.stabilityFactor
  
  public var warmupState: WarmupState {
    warmedLearningRate >= targetLearningRate ? .complete : .warming
  }
  
  let targetLearningRate: Tensor.Scalar
  let warmupSteps: Tensor.Scalar
  var globalSteps: Tensor.Scalar = 0
  
  public init(targetLearningRate: Tensor.Scalar,
              warmupSteps: Tensor.Scalar) {
    self.targetLearningRate = targetLearningRate
    self.warmupSteps = warmupSteps
  }
  
  open func reset() {
    warmedLearningRate = Tensor.Scalar.stabilityFactor
    globalSteps = 0
  }
  
  open func step() {
    // override and apply function here
    globalSteps += 1
  }
}

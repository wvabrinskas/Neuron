//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift

public protocol DecayFunction {
  var decayedLearningRate: Tensor.Scalar { get }
  func reset()
  func step()
}

open class BaseDecayFunction: DecayFunction {
  public var decayedLearningRate: Tensor.Scalar
  
  let originalLearningRate: Tensor.Scalar
  let decayRate: Tensor.Scalar
  let decaySteps: Tensor.Scalar
  var globalSteps: Tensor.Scalar = 0
  let staircase: Bool
  
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
  
  open func reset() {
    decayedLearningRate = originalLearningRate
    globalSteps = 0
  }
  
  open func step() {
    // override and apply function here
    globalSteps += 1
  }
}

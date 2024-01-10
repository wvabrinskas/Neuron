//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift

public protocol DecayFunction {
  var decayedLearningRate: Float { get }
  func reset()
  func step()
}

open class BaseDecayFunction: DecayFunction {
  public var decayedLearningRate: Float
  
  let originalLearningRate: Float
  let decayRate: Float
  let decaySteps: Float
  var globalSteps: Float = 0
  let staircase: Bool
  
  public init(learningRate: Float,
              decayRate: Float,
              decaySteps: Float,
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

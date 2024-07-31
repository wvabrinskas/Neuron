//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import NumSwift

public protocol DecayFunction {
  associatedtype N: TensorNumeric
  var decayedLearningRate: Tensor<N>.Scalar { get }
  func reset()
  func step()
}

open class BaseDecayFunction<N: TensorNumeric>: DecayFunction {
  public typealias N = N
  public var decayedLearningRate: Tensor<N>.Scalar
  
  let originalLearningRate: Tensor<N>.Scalar
  let decayRate: Tensor<N>.Scalar
  let decaySteps: Tensor<N>.Scalar
  var globalSteps: Tensor<N>.Scalar = 0
  let staircase: Bool
  
  public init(learningRate: Tensor<N>.Scalar,
              decayRate: Tensor<N>.Scalar,
              decaySteps: Tensor<N>.Scalar,
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

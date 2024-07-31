//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/9/24.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

final class DecayFunctionTests: XCTestCase {
  typealias N = Float

  func test_exponentialDecay() {
    let learningRate: Tensor<N>.Scalar = 0.001
    let steps = 10
    let exp = ExponentialDecay(learningRate: learningRate,
                               decayRate: 0.96,
                               decaySteps: 5,
                               staircase: true)
    
    for _ in 0..<steps {
      exp.step()
    }
    
        
    XCTAssertEqual(exp.decayedLearningRate, 0.00096000003, accuracy: 0.0004)
    XCTAssertEqual(exp.globalSteps, Tensor<N>.Scalar(steps))
    XCTAssertNotEqual(exp.decayedLearningRate, learningRate)
  }
}

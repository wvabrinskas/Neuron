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
  
  func test_exponentialDecay() {
    let learningRate: Tensor.Scalar = 0.001
    let steps = 10
    let exp = ExponentialDecay(learningRate: learningRate,
                               decayRate: 0.96,
                               decaySteps: 5,
                               staircase: true)
    
    for _ in 0..<steps {
      exp.step()
    }
    
        
    XCTAssertEqual(exp.decayedLearningRate, 0.00096000003, accuracy: 0.0004)
    XCTAssertEqual(exp.globalSteps, Tensor.Scalar(steps))
    XCTAssertNotEqual(exp.decayedLearningRate, learningRate)
  }
  
  func test_cosineAnnealing_initialValue() {
    let maxLR: Tensor.Scalar = 0.1
    let minLR: Tensor.Scalar = 0.001
    let epochs = 10
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           decaySteps: epochs)
    
    // Before stepping, should be at initial learning rate
    XCTAssertEqual(cosineDecay.decayedLearningRate, maxLR)
    XCTAssertEqual(cosineDecay.globalSteps, 0)
  }
  
  func test_cosineAnnealing_firstEpoch() {
    let maxLR: Tensor.Scalar = 0.1
    let minLR: Tensor.Scalar = 0.001
    let epochs = 10
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           decaySteps: epochs)
    
    cosineDecay.step()
    
    // At epoch 0: cos(π * 0 / 10) = cos(0) = 1
    // LR = 0.001 + 0.5 * (0.1 - 0.001) * (1 + 1) = 0.001 + 0.5 * 0.099 * 2 = 0.1
    XCTAssertEqual(cosineDecay.decayedLearningRate, maxLR, accuracy: 0.0001)
    XCTAssertEqual(cosineDecay.globalSteps, 1)
  }
  
  func test_cosineAnnealing_progression() {
    let maxLR: Tensor.Scalar = 1.0
    let minLR: Tensor.Scalar = 0.0
    let epochs = 4
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           decaySteps: epochs)
    
    // Track learning rate progression through epochs
    var learningRates: [Tensor.Scalar] = []
    
    for epoch in 0...epochs {
      cosineDecay.step()
      learningRates.append(cosineDecay.decayedLearningRate)
    }
    
    // Verify monotonic decrease
    for i in 1..<learningRates.count {
      XCTAssertLessThanOrEqual(learningRates[i], learningRates[i-1],
                               "Learning rate should decrease monotonically")
    }
    
    // First value should be near max
    XCTAssertEqual(learningRates[0], maxLR, accuracy: 0.0001)
    
    // Last value should be near min
    XCTAssertEqual(learningRates[epochs], minLR, accuracy: 0.0001)
  }

  func test_cosineAnnealing_reset() {
    let maxLR: Tensor.Scalar = 0.1
    let minLR: Tensor.Scalar = 0.001
    let epochs = 10
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           decaySteps: epochs)
    
    // Step through a few epochs
    for _ in 0..<5 {
      cosineDecay.step()
    }
    
    XCTAssertNotEqual(cosineDecay.decayedLearningRate, maxLR)
    XCTAssertEqual(cosineDecay.globalSteps, 5)
    
    // Reset
    cosineDecay.reset()
    
    // Should return to initial state
    XCTAssertEqual(cosineDecay.decayedLearningRate, maxLR)
    XCTAssertEqual(cosineDecay.globalSteps, 0)
  }
  
  func test_cosineAnnealing_smallRange() {
    let maxLR: Tensor.Scalar = 0.001
    let minLR: Tensor.Scalar = 0.0001
    let epochs = 100
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           decaySteps: epochs)
    
    for _ in 0..<50 {
      cosineDecay.step()
    }
    
    // At halfway point, should be at midpoint
    let expected = (maxLR + minLR) / 2
    XCTAssertEqual(cosineDecay.decayedLearningRate, expected, accuracy: 0.0001)
  }
}

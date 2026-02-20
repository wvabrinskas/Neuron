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
      exp.step(type: .batch)
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
                                           epochs: epochs)
    
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
                                           epochs: epochs)
    
    cosineDecay.step(type: .epoch(0))
    
    // At epoch 0: cos(π * 0 / 10) = cos(0) = 1
    // LR = 0.001 + 0.5 * (0.1 - 0.001) * (1 + 1) = 0.001 + 0.5 * 0.099 * 2 = 0.1
    XCTAssertEqual(cosineDecay.decayedLearningRate, maxLR, accuracy: 0.0001)
    XCTAssertEqual(cosineDecay.globalSteps, 1)
  }
  
  func test_cosineAnnealing_halfwayPoint() {
    let maxLR: Tensor.Scalar = 0.1
    let minLR: Tensor.Scalar = 0.001
    let epochs = 10
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           epochs: epochs)
    
    cosineDecay.step(type: .epoch(5))
    
    // At epoch 5: cos(π * 5 / 10) = cos(π/2) = 0
    // LR = 0.001 + 0.5 * (0.1 - 0.001) * (1 + 0) = 0.001 + 0.0495 = 0.0505
    let expected: Tensor.Scalar = 0.0505
    XCTAssertEqual(cosineDecay.decayedLearningRate, expected, accuracy: 0.0001)
  }
  
  func test_cosineAnnealing_finalEpoch() {
    let maxLR: Tensor.Scalar = 0.1
    let minLR: Tensor.Scalar = 0.001
    let epochs = 10
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           epochs: epochs)
    
    cosineDecay.step(type: .epoch(10))
    
    // At epoch 10: cos(π * 10 / 10) = cos(π) = -1
    // LR = 0.001 + 0.5 * (0.1 - 0.001) * (1 - 1) = 0.001 + 0 = 0.001
    XCTAssertEqual(cosineDecay.decayedLearningRate, minLR, accuracy: 0.0001)
  }
  
  func test_cosineAnnealing_progression() {
    let maxLR: Tensor.Scalar = 1.0
    let minLR: Tensor.Scalar = 0.0
    let epochs = 4
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           epochs: epochs)
    
    // Track learning rate progression through epochs
    var learningRates: [Tensor.Scalar] = []
    
    for epoch in 0...epochs {
      cosineDecay.step(type: .epoch(epoch))
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
  
  func test_cosineAnnealing_ignoresBatchSteps() {
    let maxLR: Tensor.Scalar = 0.1
    let minLR: Tensor.Scalar = 0.001
    let epochs = 10
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           epochs: epochs)
    
    let initialLR = cosineDecay.decayedLearningRate
    
    // Batch steps should not affect cosine annealing
    for _ in 0..<5 {
      cosineDecay.step(type: .batch)
    }
    
    // Learning rate should not change from batch steps
    XCTAssertEqual(cosineDecay.decayedLearningRate, initialLR)
    
    // Global steps should not increment for batch steps (early return in guard)
    XCTAssertEqual(cosineDecay.globalSteps, 0)
  }
  
  func test_cosineAnnealing_reset() {
    let maxLR: Tensor.Scalar = 0.1
    let minLR: Tensor.Scalar = 0.001
    let epochs = 10
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           epochs: epochs)
    
    // Step through a few epochs
    cosineDecay.step(type: .epoch(5))
    XCTAssertNotEqual(cosineDecay.decayedLearningRate, maxLR)
    XCTAssertEqual(cosineDecay.globalSteps, 1)
    
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
                                           epochs: epochs)
    
    cosineDecay.step(type: .epoch(50))
    
    // At halfway point, should be at midpoint
    let expected = (maxLR + minLR) / 2
    XCTAssertEqual(cosineDecay.decayedLearningRate, expected, accuracy: 0.00001)
  }
}

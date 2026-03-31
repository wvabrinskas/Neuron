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
  
  // MARK: - LinearDecay

  func test_linearDecay_initialValue() {
    let learningRate: Tensor.Scalar = 0.01
    let decay = LinearDecay(learningRate: learningRate, decaySteps: 100)

    XCTAssertEqual(decay.decayedLearningRate, learningRate)
    XCTAssertEqual(decay.globalSteps, 0)
  }

  func test_linearDecay_firstStep_usesGlobalStepsZero() {
    // LinearDecay computes rate from globalSteps BEFORE super.step() increments it,
    // so the first step still yields the original learning rate.
    let learningRate: Tensor.Scalar = 0.01
    let decay = LinearDecay(learningRate: learningRate, decaySteps: 100)

    decay.step()

    XCTAssertEqual(decay.decayedLearningRate, learningRate, accuracy: 1e-6)
    XCTAssertEqual(decay.globalSteps, 1)
  }

  func test_linearDecay_midpointValue() {
    // After decaySteps/2 + 1 calls to step(), globalSteps == decaySteps/2 when
    // the rate is computed, giving lr * (1 - 0.5) = lr * 0.5.
    let learningRate: Tensor.Scalar = 0.1
    let decaySteps: Tensor.Scalar = 100
    let decay = LinearDecay(learningRate: learningRate, decaySteps: decaySteps)

    for _ in 0..<Int(decaySteps / 2) + 1 {
      decay.step()
    }

    let expected = learningRate * 0.5
    XCTAssertEqual(decay.decayedLearningRate, expected, accuracy: 1e-5)
  }

  func test_linearDecay_reachesZeroAtDecaySteps() {
    // After decaySteps + 1 calls, globalSteps == decaySteps when the rate is
    // computed: lr * (1 - decaySteps/decaySteps) == 0.
    let learningRate: Tensor.Scalar = 0.01
    let decaySteps: Tensor.Scalar = 10
    let decay = LinearDecay(learningRate: learningRate, decaySteps: decaySteps)

    for _ in 0..<Int(decaySteps) + 1 {
      decay.step()
    }

    XCTAssertEqual(decay.decayedLearningRate, 0, accuracy: 1e-6)
  }

  func test_linearDecay_monotonicDecrease() {
    let learningRate: Tensor.Scalar = 0.1
    let decaySteps: Tensor.Scalar = 20
    let decay = LinearDecay(learningRate: learningRate, decaySteps: decaySteps)

    var rates: [Tensor.Scalar] = [decay.decayedLearningRate]
    for _ in 0..<Int(decaySteps) {
      decay.step()
      rates.append(decay.decayedLearningRate)
    }

    for i in 1..<rates.count {
      XCTAssertLessThanOrEqual(rates[i], rates[i - 1],
                               "Learning rate should not increase at index \(i)")
    }
  }

  func test_linearDecay_specificStep() {
    // After 6 steps: globalSteps == 5 when rate computed → lr * (1 - 5/10) = 0.5 * lr
    let learningRate: Tensor.Scalar = 0.1
    let decaySteps: Tensor.Scalar = 10
    let decay = LinearDecay(learningRate: learningRate, decaySteps: decaySteps)

    for _ in 0..<6 {
      decay.step()
    }

    let expected = learningRate * (1 - 5 / decaySteps)
    XCTAssertEqual(decay.decayedLearningRate, expected, accuracy: 1e-6)
  }

  func test_linearDecay_reset() {
    let learningRate: Tensor.Scalar = 0.01
    let decay = LinearDecay(learningRate: learningRate, decaySteps: 50)

    for _ in 0..<25 {
      decay.step()
    }

    XCTAssertNotEqual(decay.decayedLearningRate, learningRate)
    XCTAssertEqual(decay.globalSteps, 25)

    decay.reset()

    XCTAssertEqual(decay.decayedLearningRate, learningRate)
    XCTAssertEqual(decay.globalSteps, 0)
  }

  func test_linearDecay_defaultDecaySteps() {
    // Default decaySteps is 1000; after 501 steps globalSteps == 500 → lr * 0.5
    let learningRate: Tensor.Scalar = 0.1
    let decay = LinearDecay(learningRate: learningRate)

    for _ in 0..<501 {
      decay.step()
    }

    let expected = learningRate * (1 - 500.0 / 1000.0)
    XCTAssertEqual(decay.decayedLearningRate, expected, accuracy: 1e-5)
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

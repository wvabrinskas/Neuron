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
    
    // After step(), globalSteps=1: cos(π * 1 / 10) ≈ 0.9511
    // LR = 0.001 + 0.5 * (0.1 - 0.001) * (1 + 0.9511) ≈ 0.09758
    XCTAssertEqual(cosineDecay.decayedLearningRate, 0.09758, accuracy: 0.0001)
    XCTAssertEqual(cosineDecay.globalSteps, 1)
  }
  
  func test_cosineAnnealing_progression() {
    let maxLR: Tensor.Scalar = 1.0
    let minLR: Tensor.Scalar = 0.0
    let epochs = 4
    
    let cosineDecay = CosineAnnealingDecay(learningRate: maxLR,
                                           minLearningRate: minLR,
                                           decaySteps: epochs)
    
    // Collect initial value (maxLR), then step epochs times.
    // At globalSteps=epochs: cos(π)=-1 → LR = minLR.
    var learningRates: [Tensor.Scalar] = [cosineDecay.decayedLearningRate]

    for _ in 0..<epochs {
      cosineDecay.step()
      learningRates.append(cosineDecay.decayedLearningRate)
    }

    // Verify monotonic decrease
    for i in 1..<learningRates.count {
      XCTAssertLessThanOrEqual(learningRates[i], learningRates[i-1],
                               "Learning rate should decrease monotonically")
    }

    // First value should be near max (initial, before any step)
    XCTAssertEqual(learningRates[0], maxLR, accuracy: 0.0001)

    // Last value should be near min (after epochs steps, globalSteps==epochs)
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

  func test_linearDecay_firstStep_usesGlobalStepsOne() {
    // LinearDecay increments globalSteps first, then computes:
    // After 1 step: globalSteps=1, rate = learningRate * (1 - 1/decaySteps)
    let learningRate: Tensor.Scalar = 0.01
    let decay = LinearDecay(learningRate: learningRate, decaySteps: 100)

    decay.step()

    XCTAssertEqual(decay.decayedLearningRate, learningRate * (1 - 1 / 100), accuracy: 1e-6)
    XCTAssertEqual(decay.globalSteps, 1)
  }

  func test_linearDecay_midpointValue() {
    // After decaySteps/2 steps, globalSteps=decaySteps/2:
    // lr * (1 - (decaySteps/2) / decaySteps) = lr * 0.5
    let learningRate: Tensor.Scalar = 0.1
    let decaySteps: Tensor.Scalar = 100
    let decay = LinearDecay(learningRate: learningRate, decaySteps: decaySteps)

    for _ in 0..<Int(decaySteps / 2) {
      decay.step()
    }

    let expected = learningRate * 0.5
    XCTAssertEqual(decay.decayedLearningRate, expected, accuracy: 1e-5)
  }

  func test_linearDecay_reachesZeroAtDecaySteps() {
    // After decaySteps steps, globalSteps=decaySteps:
    // lr * (1 - decaySteps/decaySteps) == 0
    let learningRate: Tensor.Scalar = 0.01
    let decaySteps: Tensor.Scalar = 10
    let decay = LinearDecay(learningRate: learningRate, decaySteps: decaySteps)

    for _ in 0..<Int(decaySteps) {
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
    // After 6 steps: globalSteps=6 → lr * (1 - 6/10) = 0.4 * lr
    let learningRate: Tensor.Scalar = 0.1
    let decaySteps: Tensor.Scalar = 10
    let decay = LinearDecay(learningRate: learningRate, decaySteps: decaySteps)

    for _ in 0..<6 {
      decay.step()
    }

    let expected = learningRate * (1 - 6 / decaySteps)
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
    // Default decaySteps is 1000; after 500 steps globalSteps=500 → lr * 0.5
    let learningRate: Tensor.Scalar = 0.1
    let decay = LinearDecay(learningRate: learningRate)

    for _ in 0..<500 {
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

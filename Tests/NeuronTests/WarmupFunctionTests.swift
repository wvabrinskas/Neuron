//
//  WarmupFunctionTests.swift
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

final class WarmupFunctionTests: XCTestCase {

  // MARK: - BaseWarmupFunction

  func test_baseWarmupFunction_initialState() {
    let warmup = LinearWarmupFunction(targetLearningRate: 0.01, warmupSteps: 10)
    XCTAssertEqual(warmup.warmedLearningRate, Tensor.Scalar.stabilityFactor)
    XCTAssertEqual(warmup.warmupState, .warming)
  }

  func test_baseWarmupFunction_reset() {
    let warmup = LinearWarmupFunction(targetLearningRate: 0.01, warmupSteps: 5)

    for _ in 0..<3 {
      warmup.step()
    }

    XCTAssertNotEqual(warmup.warmedLearningRate, Tensor.Scalar.stabilityFactor)
    XCTAssertEqual(warmup.globalSteps, 3)

    warmup.reset()

    XCTAssertEqual(warmup.warmedLearningRate, 0)
    XCTAssertEqual(warmup.globalSteps, 0)
    XCTAssertEqual(warmup.warmupState, .warming)
  }

  // MARK: - LinearWarmupFunction

  func test_linear_firstStep_isZero() {
    let targetLR: Tensor.Scalar = 0.01
    let warmupSteps: Tensor.Scalar = 10
    let warmup = LinearWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    warmup.step()

    // globalSteps was 0 before step, so newLr = targetLR * 0 / warmupSteps = 0
    XCTAssertEqual(warmup.warmedLearningRate, 0, accuracy: 1e-7)
    XCTAssertEqual(warmup.globalSteps, 1)
    XCTAssertEqual(warmup.warmupState, .warming)
  }

  func test_linear_progression() {
    let targetLR: Tensor.Scalar = 1.0
    let warmupSteps: Tensor.Scalar = 4
    let warmup = LinearWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    // step() uses globalSteps BEFORE incrementing
    // Step 1: globalSteps=0 → 0/4 = 0.0
    // Step 2: globalSteps=1 → 1/4 = 0.25
    // Step 3: globalSteps=2 → 2/4 = 0.5
    // Step 4: globalSteps=3 → 3/4 = 0.75
    // Step 5: globalSteps=4 → 4/4 = 1.0 (target reached)
    let expected: [Tensor.Scalar] = [0.0, 0.25, 0.5, 0.75, 1.0]

    for i in 0..<Int(warmupSteps) + 1 {
      warmup.step()
      XCTAssertEqual(warmup.warmedLearningRate, expected[i], accuracy: 1e-5,
                     "Mismatch at step \(i + 1)")
    }
  }

  func test_linear_reachesTargetAfterWarmupSteps() {
    let targetLR: Tensor.Scalar = 0.001
    let warmupSteps: Tensor.Scalar = 10
    let warmup = LinearWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    // warmupSteps + 1 calls to step() brings globalSteps to warmupSteps
    for _ in 0..<Int(warmupSteps) + 1 {
      warmup.step()
    }

    XCTAssertEqual(warmup.warmedLearningRate, targetLR, accuracy: 1e-7)
    XCTAssertEqual(warmup.warmupState, .complete)
  }

  func test_linear_warmupStateTransition() {
    let targetLR: Tensor.Scalar = 0.01
    let warmupSteps: Tensor.Scalar = 3
    let warmup = LinearWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    XCTAssertEqual(warmup.warmupState, .warming)

    // Step through until complete
    for _ in 0..<Int(warmupSteps) {
      warmup.step()
      XCTAssertEqual(warmup.warmupState, .warming)
    }

    warmup.step() // globalSteps == warmupSteps → warmedLR == targetLR
    XCTAssertEqual(warmup.warmupState, .complete)
  }

  func test_linear_monotonicallyIncreasing() {
    let targetLR: Tensor.Scalar = 0.05
    let warmupSteps: Tensor.Scalar = 20
    let warmup = LinearWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    var previous: Tensor.Scalar = -Tensor.Scalar.infinity
    for i in 0..<Int(warmupSteps) + 1 {
      warmup.step()
      XCTAssertGreaterThanOrEqual(warmup.warmedLearningRate, previous,
                                  "LR should be non-decreasing at step \(i + 1)")
      previous = warmup.warmedLearningRate
    }
  }

  func test_linear_reset_allowsReuse() {
    let targetLR: Tensor.Scalar = 0.01
    let warmupSteps: Tensor.Scalar = 5
    let warmup = LinearWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    for _ in 0..<Int(warmupSteps) + 1 {
      warmup.step()
    }
    XCTAssertEqual(warmup.warmupState, .complete)

    warmup.reset()
    XCTAssertEqual(warmup.warmupState, .warming)
    XCTAssertEqual(warmup.globalSteps, 0)

    // Should produce the same progression after reset
    warmup.step()
    XCTAssertEqual(warmup.warmedLearningRate, 0, accuracy: 1e-7)
  }

  // MARK: - CosineWarmupFunction

  func test_cosine_firstStep_isZero() {
    let targetLR: Tensor.Scalar = 0.01
    let warmup = CosineWarmupFunction(targetLearningRate: targetLR, warmupSteps: 10)

    warmup.step()

    // globalSteps=0 → targetLR * 0.5 * (1 - cos(0)) = targetLR * 0.5 * 0 = 0
    XCTAssertEqual(warmup.warmedLearningRate, 0, accuracy: 1e-6)
    XCTAssertEqual(warmup.globalSteps, 1)
  }

  func test_cosine_reachesTargetAfterWarmupSteps() {
    let targetLR: Tensor.Scalar = 0.001
    let warmupSteps: Tensor.Scalar = 10
    let warmup = CosineWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    // At globalSteps == warmupSteps: 0.5 * (1 - cos(π)) = 0.5 * 2 = 1.0
    for _ in 0..<Int(warmupSteps) + 1 {
      warmup.step()
    }

    XCTAssertEqual(warmup.warmedLearningRate, targetLR, accuracy: 1e-6)
    XCTAssertEqual(warmup.warmupState, .complete)
  }

  func test_cosine_halfwayIsHalfTarget() {
    let targetLR: Tensor.Scalar = 1.0
    let warmupSteps: Tensor.Scalar = 10
    let warmup = CosineWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    // At globalSteps == warmupSteps/2: cos(π/2) = 0 → 0.5 * (1 - 0) = 0.5
    for _ in 0..<Int(warmupSteps) / 2 + 1 {
      warmup.step()
    }

    XCTAssertEqual(warmup.warmedLearningRate, targetLR * 0.5, accuracy: 1e-5)
  }

  func test_cosine_monotonicallyIncreasing() {
    let targetLR: Tensor.Scalar = 0.1
    let warmupSteps: Tensor.Scalar = 20
    let warmup = CosineWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    var previous: Tensor.Scalar = -Tensor.Scalar.infinity
    for i in 0..<Int(warmupSteps) + 1 {
      warmup.step()
      XCTAssertGreaterThanOrEqual(warmup.warmedLearningRate, previous,
                                  "Cosine LR should be non-decreasing at step \(i + 1)")
      previous = warmup.warmedLearningRate
    }
  }

  func test_cosine_warmupStateTransition() {
    let targetLR: Tensor.Scalar = 0.01
    let warmupSteps: Tensor.Scalar = 4
    let warmup = CosineWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    XCTAssertEqual(warmup.warmupState, .warming)

    for _ in 0..<Int(warmupSteps) {
      warmup.step()
      XCTAssertEqual(warmup.warmupState, .warming)
    }

    warmup.step() // globalSteps == warmupSteps → warmedLR == targetLR
    XCTAssertEqual(warmup.warmupState, .complete)
  }

  func test_cosine_reset() {
    let targetLR: Tensor.Scalar = 0.01
    let warmupSteps: Tensor.Scalar = 5
    let warmup = CosineWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    for _ in 0..<Int(warmupSteps) + 1 {
      warmup.step()
    }
    XCTAssertEqual(warmup.warmupState, .complete)

    warmup.reset()

    XCTAssertEqual(warmup.warmedLearningRate, 0)
    XCTAssertEqual(warmup.globalSteps, 0)
    XCTAssertEqual(warmup.warmupState, .warming)
  }

  func test_cosine_curveShape() {
    // Verify the cosine curve accelerates faster than linear initially
    // At step globalSteps=1 out of warmupSteps=4:
    // Cosine: 0.5 * (1 - cos(π/4)) ≈ 0.5 * (1 - 0.707) ≈ 0.146
    // Linear: 1/4 = 0.25
    // → cosine is slower initially then faster
    let targetLR: Tensor.Scalar = 1.0
    let warmupSteps: Tensor.Scalar = 4
    let cosine = CosineWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)
    let linear = LinearWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)

    cosine.step(); cosine.step() // globalSteps was 1 when computing
    linear.step(); linear.step()

    // Both should be strictly between 0 and targetLR at this point
    XCTAssertGreaterThan(cosine.warmedLearningRate, 0)
    XCTAssertLessThan(cosine.warmedLearningRate, targetLR)
    XCTAssertGreaterThan(linear.warmedLearningRate, 0)
    XCTAssertLessThan(linear.warmedLearningRate, targetLR)
  }
}

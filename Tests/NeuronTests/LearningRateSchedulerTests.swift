//
//  LearningRateSchedulerTests.swift
//
//  Created by William Vabrinskas on 3/31/26.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

final class LearningRateSchedulerTests: XCTestCase {

  // MARK: - Helpers

  private func makeScheduler(
    initialLR: Tensor.Scalar = 0.0001,
    targetLR: Tensor.Scalar = 0.1,
    warmupSteps: Tensor.Scalar = 4,
    decayRate: Tensor.Scalar = 0.5,
    decaySteps: Tensor.Scalar = 1,
    type: LearningRateScheduleStepType = .batch
  ) -> (SequentialLearningRateScheduler, LinearWarmupFunction, ExponentialDecay) {
    let warmup = LinearWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)
    let decay = ExponentialDecay(learningRate: targetLR, decayRate: decayRate, decaySteps: decaySteps)
    let scheduler = SequentialLearningRateScheduler(learningRate: initialLR, warmup: warmup, decay: decay, type: type)
    return (scheduler, warmup, decay)
  }

  // MARK: - Initial State

  func test_initialLearningRate_equalsPassedValue() {
    let (scheduler, _, _) = makeScheduler(initialLR: 0.0005)
    XCTAssertEqual(scheduler.learningRate, 0.0005)
  }

  // MARK: - Step Type Guard

  func test_step_wrongType_doesNothing_batchScheduler() {
    let (scheduler, _, _) = makeScheduler(initialLR: 0.0001, type: .batch)
    scheduler.step(type: .epoch)
    XCTAssertEqual(scheduler.learningRate, 0.0001)
  }

  func test_step_wrongType_doesNothing_epochScheduler() {
    let (scheduler, _, _) = makeScheduler(initialLR: 0.0001, type: .epoch)
    scheduler.step(type: .batch)
    XCTAssertEqual(scheduler.learningRate, 0.0001)
  }

  func test_step_correctEpochType_advancesScheduler() {
    let (scheduler, _, _) = makeScheduler(initialLR: 0.0001, type: .epoch)
    scheduler.step(type: .epoch)
    // First warmup step: globalSteps=0 → warmedLR = targetLR * 0/warmupSteps = 0
    XCTAssertEqual(scheduler.learningRate, 0, accuracy: 1e-7)
  }

  // MARK: - Warmup Phase

  func test_warmupPhase_learningRateFollowsWarmup() {
    let (scheduler, warmup, _) = makeScheduler(targetLR: 1.0, warmupSteps: 4)

    for _ in 0..<4 {
      scheduler.step(type: .batch)
      XCTAssertEqual(scheduler.learningRate, warmup.warmedLearningRate, accuracy: 1e-6)
    }
  }

  func test_warmupPhase_linearProgression() {
    // warmupSteps=4, targetLR=1.0
    // Step 1: globalSteps=0 → 0/4 = 0.0
    // Step 2: globalSteps=1 → 1/4 = 0.25
    // Step 3: globalSteps=2 → 2/4 = 0.5
    // Step 4: globalSteps=3 → 3/4 = 0.75
    let (scheduler, _, _) = makeScheduler(targetLR: 1.0, warmupSteps: 4)
    let expected: [Tensor.Scalar] = [0.0, 0.25, 0.5, 0.75]

    for i in 0..<4 {
      scheduler.step(type: .batch)
      XCTAssertEqual(scheduler.learningRate, expected[i], accuracy: 1e-5,
                     "Mismatch at warmup step \(i + 1)")
    }
  }

  func test_warmupPhase_doesNotCallDecay() {
    let (scheduler, _, decay) = makeScheduler(targetLR: 0.1, warmupSteps: 5)
    let initialDecayLR = decay.decayedLearningRate

    // Step through entire warmup (warmupSteps steps, state not yet .complete)
    for _ in 0..<5 {
      scheduler.step(type: .batch)
    }

    // Decay should not have been touched yet
    XCTAssertEqual(decay.decayedLearningRate, initialDecayLR)
  }

  // MARK: - Warmup → Decay Transition

  func test_transitionToDecay_afterWarmupCompletes() {
    let targetLR: Tensor.Scalar = 0.1
    let warmupSteps: Tensor.Scalar = 3
    let (scheduler, _, decay) = makeScheduler(targetLR: targetLR, warmupSteps: warmupSteps, decayRate: 0.5, decaySteps: 1)

    // Complete warmup: warmupSteps+1 calls to reach .complete
    for _ in 0..<Int(warmupSteps) + 1 {
      scheduler.step(type: .batch)
    }
    XCTAssertEqual(scheduler.learningRate, targetLR, accuracy: 1e-6)

    // Next step enters decay phase
    scheduler.step(type: .batch)
    XCTAssertEqual(scheduler.learningRate, decay.decayedLearningRate, accuracy: 1e-6)
  }

  func test_decayPhase_learningRateFollowsDecay() {
    let warmupSteps: Tensor.Scalar = 2
    let (scheduler, _, decay) = makeScheduler(targetLR: 1.0, warmupSteps: warmupSteps, decayRate: 0.5, decaySteps: 1)

    // Exhaust warmup
    for _ in 0..<Int(warmupSteps) + 1 {
      scheduler.step(type: .batch)
    }

    // Subsequent steps should track decay
    for _ in 0..<5 {
      scheduler.step(type: .batch)
      XCTAssertEqual(scheduler.learningRate, decay.decayedLearningRate, accuracy: 1e-6)
    }
  }

  func test_decayPhase_learningRateDecreases() {
    let warmupSteps: Tensor.Scalar = 3
    let (scheduler, _, _) = makeScheduler(targetLR: 1.0, warmupSteps: warmupSteps, decayRate: 0.5, decaySteps: 1)

    // Exhaust warmup
    for _ in 0..<Int(warmupSteps) + 1 {
      scheduler.step(type: .batch)
    }

    var previousLR = scheduler.learningRate
    for i in 0..<5 {
      scheduler.step(type: .batch)
      XCTAssertLessThanOrEqual(scheduler.learningRate, previousLR,
                               "Decay LR should be non-increasing at decay step \(i + 1)")
      previousLR = scheduler.learningRate
    }
  }

  // MARK: - Reset

  func test_reset_restoresWarmupLearningRate() {
    let (scheduler, _, _) = makeScheduler()

    for _ in 0..<10 {
      scheduler.step(type: .batch)
    }

    scheduler.reset()

    // After reset, learningRate is set from warmup.warmedLearningRate = stabilityFactor
    XCTAssertEqual(scheduler.learningRate, Tensor.Scalar.stabilityFactor)
  }

  func test_reset_resetsWarmupState() {
    let (scheduler, warmup, _) = makeScheduler(warmupSteps: 3)

    // Complete warmup
    for _ in 0..<4 {
      scheduler.step(type: .batch)
    }
    XCTAssertEqual(warmup.warmupState, .complete)

    scheduler.reset()

    XCTAssertEqual(warmup.warmupState, .warming)
    XCTAssertEqual(warmup.globalSteps, 0)
  }

  func test_reset_resetsDecayState() {
    let targetLR: Tensor.Scalar = 0.1
    let warmupSteps: Tensor.Scalar = 2
    let (scheduler, _, decay) = makeScheduler(targetLR: targetLR, warmupSteps: warmupSteps)

    // Run through warmup and some decay
    for _ in 0..<10 {
      scheduler.step(type: .batch)
    }

    scheduler.reset()

    XCTAssertEqual(decay.decayedLearningRate, targetLR)
  }

  func test_reset_allowsWarmupToRunAgain() {
    let targetLR: Tensor.Scalar = 1.0
    let warmupSteps: Tensor.Scalar = 3
    let (scheduler, _, _) = makeScheduler(targetLR: targetLR, warmupSteps: warmupSteps)

    // Complete first warmup cycle
    for _ in 0..<Int(warmupSteps) + 1 {
      scheduler.step(type: .batch)
    }
    XCTAssertEqual(scheduler.learningRate, targetLR, accuracy: 1e-6)

    scheduler.reset()

    // First step after reset should restart warmup from 0
    scheduler.step(type: .batch)
    XCTAssertEqual(scheduler.learningRate, 0, accuracy: 1e-7)
  }

  // MARK: - Cosine Warmup Integration

  func test_cosineWarmup_progression() {
    let targetLR: Tensor.Scalar = 1.0
    let warmupSteps: Tensor.Scalar = 10
    let warmup = CosineWarmupFunction(targetLearningRate: targetLR, warmupSteps: warmupSteps)
    let decay = ExponentialDecay(learningRate: targetLR, decayRate: 0.99, decaySteps: 100)
    let scheduler = SequentialLearningRateScheduler(learningRate: 0.0001, warmup: warmup, decay: decay, type: .batch)

    var previous: Tensor.Scalar = -Tensor.Scalar.infinity
    for i in 0..<Int(warmupSteps) + 1 {
      scheduler.step(type: .batch)
      XCTAssertGreaterThanOrEqual(scheduler.learningRate, previous,
                                  "Cosine warmup LR should be non-decreasing at step \(i + 1)")
      previous = scheduler.learningRate
    }

    XCTAssertEqual(scheduler.learningRate, targetLR, accuracy: 1e-5)
  }
}

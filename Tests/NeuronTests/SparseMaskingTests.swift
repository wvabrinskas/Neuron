//
//  SparseMaskingTests.swift
//
//  Covers ignore-index masking on the sparse losses and the accuracy calculator.
//

import XCTest
@testable import Neuron

final class SparseMaskingTests: XCTestCase {

  /// Three timesteps over a 3-class vocabulary; the model is confident and correct at t0/t1
  /// and confidently wrong at t2, which the label marks as padding (class 0).
  private func fixture() -> (predicted: Tensor, correct: Tensor) {
    let predicted = Tensor([[[0.1, 0.8, 0.1]],
                            [[0.1, 0.1, 0.8]],
                            [[0.8, 0.1, 0.1]]])
    let correct = Tensor([[[1]], [[2]], [[0]]])
    return (predicted, correct)
  }

  // MARK: - Loss

  func test_calculate_ignoredPositionContributesNoLoss() {
    let (predicted, correct) = fixture()

    let masked = LossFunction.sparseCrossEntropySoftmax.calculate(predicted, correct: correct, ignoring: 0)

    // The padded timestep is the third depth slice of the loss tensor.
    XCTAssertEqual(masked.storage[2], 0, accuracy: 1e-6)
    XCTAssertGreaterThan(masked.storage[0], 0)
    XCTAssertGreaterThan(masked.storage[1], 0)
  }

  func test_calculate_withoutIgnoreIndex_isUnchanged() {
    let (predicted, correct) = fixture()

    let a = LossFunction.sparseCrossEntropySoftmax.calculate(predicted, correct: correct)
    let b = LossFunction.sparseCrossEntropySoftmax.calculate(predicted, correct: correct, ignoring: nil)

    XCTAssertEqual(a.storage.count, b.storage.count)
    for i in 0..<a.storage.count {
      XCTAssertEqual(a.storage[i], b.storage[i], accuracy: 1e-6)
    }
  }

  func test_calculate_averagesOverScoredPositionsOnly() {
    let (predicted, correct) = fixture()

    let masked = LossFunction.sparseCrossEntropySoftmax.calculate(predicted, correct: correct, ignoring: 0)
    let unmasked = LossFunction.sparseCrossEntropySoftmax.calculate(predicted, correct: correct)

    // Same numerator at t0, divided by 2 scored positions instead of 3 total.
    XCTAssertEqual(masked.storage[0], unmasked.storage[0] * (3.0 / 2.0), accuracy: 1e-5)
  }

  func test_calculate_allPositionsIgnored_isZero() {
    let predicted = Tensor([[[0.1, 0.8, 0.1]], [[0.1, 0.1, 0.8]]])
    let correct = Tensor([[[0]], [[0]]])

    let loss = LossFunction.sparseCrossEntropySoftmax.calculate(predicted, correct: correct, ignoring: 0)

    XCTAssertEqual(loss.storage.reduce(0, +), 0, accuracy: 1e-6)
  }

  // MARK: - Derivative

  func test_derivative_ignoredPositionIsZeroRow() {
    let (predicted, correct) = fixture()

    let d = LossFunction.sparseCrossEntropySoftmax.derivative(predicted, correct: correct, ignoring: 0)

    // Third depth slice = flat indices 6..8. Leaving `predicted` there would drag every
    // logit down at a timestep the model should not be trained on at all.
    for i in 6..<9 {
      XCTAssertEqual(d.storage[i], 0, accuracy: 1e-6)
    }
    XCTAssertNotEqual(d.storage[1], 0)
  }

  func test_derivative_sparseCrossEntropy_ignoredPositionIsZeroRow() {
    let (predicted, correct) = fixture()

    let d = LossFunction.sparseCrossEntropy.derivative(predicted, correct: correct, ignoring: 0)

    for i in 6..<9 {
      XCTAssertEqual(d.storage[i], 0, accuracy: 1e-6)
    }
    XCTAssertNotEqual(d.storage[1], 0)
  }

  func test_derivative_withoutIgnoreIndex_isUnchanged() {
    let (predicted, correct) = fixture()

    let a = LossFunction.sparseCrossEntropySoftmax.derivative(predicted, correct: correct)
    let b = LossFunction.sparseCrossEntropySoftmax.derivative(predicted, correct: correct, ignoring: nil)

    for i in 0..<a.storage.count {
      XCTAssertEqual(a.storage[i], b.storage[i], accuracy: 1e-6)
    }
  }

  // MARK: - Accuracy

  func test_accuracy_ignoredPositionsAreNotScored() {
    let (predicted, correct) = fixture()
    let reporter = MetricsReporter(frequency: 1, metricsToGather: [.accuracy])

    // Unmasked: 2 of 3 correct, because the padded timestep is "correctly" predicted as class 0.
    let unmasked = reporter.calculateAccuracy(predicted, label: correct, binary: false, sparse: true)
    XCTAssertEqual(unmasked, 100.0, accuracy: 1e-4)

    // Masked: only t0 and t1 are scored, both correct.
    let reporter2 = MetricsReporter(frequency: 1, metricsToGather: [.accuracy])
    let masked = reporter2.calculateAccuracy(predicted, label: correct, binary: false, sparse: true, ignoring: 0)
    XCTAssertEqual(masked, 100.0, accuracy: 1e-4)
  }

  func test_accuracy_maskingRemovesFreeCorrectness() {
    // t0 wrong, t1 padding that the model trivially gets right.
    let predicted = Tensor([[[0.8, 0.1, 0.1]],
                            [[0.8, 0.1, 0.1]]])
    let correct = Tensor([[[1]], [[0]]])

    let unmasked = MetricsReporter(frequency: 1, metricsToGather: [.accuracy])
      .calculateAccuracy(predicted, label: correct, binary: false, sparse: true)
    let masked = MetricsReporter(frequency: 1, metricsToGather: [.accuracy])
      .calculateAccuracy(predicted, label: correct, binary: false, sparse: true, ignoring: 0)

    // The pad timestep is free credit: 50% without masking, 0% with it.
    XCTAssertEqual(unmasked, 50.0, accuracy: 1e-4)
    XCTAssertEqual(masked, 0.0, accuracy: 1e-4)
  }

  func test_accuracy_allPositionsIgnored_isZeroNotNaN() {
    let predicted = Tensor([[[0.8, 0.1, 0.1]], [[0.8, 0.1, 0.1]]])
    let correct = Tensor([[[0]], [[0]]])

    let accuracy = MetricsReporter(frequency: 1, metricsToGather: [.accuracy])
      .calculateAccuracy(predicted, label: correct, binary: false, sparse: true, ignoring: 0)

    XCTAssertFalse(accuracy.isNaN)
    XCTAssertEqual(accuracy, 0.0, accuracy: 1e-4)
  }
}

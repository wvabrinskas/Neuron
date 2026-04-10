//
//  LossFunctionTests.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/11/26.
//

@testable import Neuron
import Foundation
import XCTest


final class LossFunctionTests: XCTestCase {

  private let accuracy: Tensor.Scalar = 0.0001

  // MARK: - Mean Square Error

  func test_meanSquareError_calculateScalar() {
    // ((0.5-1)^2 + (0.3-0)^2 + (0.2-0)^2) / 3 = 0.38 / 3 ≈ 0.12667
    let predicted: [Tensor.Scalar] = [0.5, 0.3, 0.2]
    let correct: [Tensor.Scalar] = [1.0, 0.0, 0.0]
    let loss = LossFunction.meanSquareError.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.12667, accuracy: accuracy)
  }

  func test_meanSquareError_calculateTensor() {
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.5, 0.3, 0.2], size: size)
    let correct = Tensor([1.0, 0.0, 0.0], size: size)
    let loss = LossFunction.meanSquareError.calculate(predicted, correct: correct)
    XCTAssertEqual(loss.asScalar(), 0.12667, accuracy: accuracy)
  }

  func test_meanSquareError_derivative() {
    // 2 * (predicted - correct) / N, N=3 → [-1.0/3, 0.6/3, 0.4/3]
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.5, 0.3, 0.2], size: size)
    let correct = Tensor([1.0, 0.0, 0.0], size: size)
    let derivative = LossFunction.meanSquareError.derivative(predicted, correct: correct)
    let expected = Tensor([-1.0 / 3.0, 0.6 / 3.0, 0.4 / 3.0] as Tensor.Value, size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: accuracy))
  }

  func test_meanSquareError_perfectPrediction() {
    let predicted: [Tensor.Scalar] = [1.0, 0.0, 0.0]
    let correct: [Tensor.Scalar] = [1.0, 0.0, 0.0]
    let loss = LossFunction.meanSquareError.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.0, accuracy: accuracy)
  }

  // MARK: - Cross Entropy

  func test_crossEntropy_calculateScalar() {
    // indexOfMax([0,1,0]) = 1, p = 0.7, -log(0.7) ≈ 0.35667
    let predicted: [Tensor.Scalar] = [0.1, 0.7, 0.2]
    let correct: [Tensor.Scalar] = [0.0, 1.0, 0.0]
    let loss = LossFunction.crossEntropy.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.35667, accuracy: accuracy)
  }

  func test_crossEntropy_calculateTensor() {
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.1, 0.7, 0.2], size: size)
    let correct = Tensor([0.0, 1.0, 0.0], size: size)
    let loss = LossFunction.crossEntropy.calculate(predicted, correct: correct)
    XCTAssertEqual(loss.asScalar(), 0.35667, accuracy: accuracy)
  }

  func test_crossEntropy_derivative() {
    // -1/p for each element: [-10, -1/0.7, -5]
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.1, 0.7, 0.2], size: size)
    let correct = Tensor([0.0, 1.0, 0.0], size: size)
    let derivative = LossFunction.crossEntropy.derivative(predicted, correct: correct)
    let expected = Tensor([-10.0, -1.0 / 0.7, -5.0] as Tensor.Value, size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: accuracy))
  }

  // MARK: - Cross Entropy Softmax

  func test_crossEntropySoftmax_calculateScalar() {
    // Same forward calculation as crossEntropy
    let predicted: [Tensor.Scalar] = [0.1, 0.7, 0.2]
    let correct: [Tensor.Scalar] = [0.0, 1.0, 0.0]
    let loss = LossFunction.crossEntropySoftmax.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.35667, accuracy: accuracy)
  }

  func test_crossEntropySoftmax_calculateTensor() {
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.1, 0.7, 0.2], size: size)
    let correct = Tensor([0.0, 1.0, 0.0], size: size)
    let loss = LossFunction.crossEntropySoftmax.calculate(predicted, correct: correct)
    XCTAssertEqual(loss.asScalar(), 0.35667, accuracy: accuracy)
  }

  func test_crossEntropySoftmax_derivative() {
    // predicted - correct = [0.1-0, 0.7-1, 0.2-0] = [0.1, -0.3, 0.2]
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.1, 0.7, 0.2], size: size)
    let correct = Tensor([0.0, 1.0, 0.0], size: size)
    let derivative = LossFunction.crossEntropySoftmax.derivative(predicted, correct: correct)
    let expected = Tensor([0.1, -0.3, 0.2], size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: accuracy))
  }

  func test_crossEntropySoftmax_calculateTensor_multiDepth() {
    // Verify per-depth losses are computed and divided by depthScalar
    let size = TensorSize(rows: 1, columns: 3, depth: 2)
    let predicted = Tensor([0.1, 0.7, 0.2,   // depth 0
                             0.2, 0.6, 0.2],  // depth 1
                           size: size)
    let correct = Tensor([0.0, 1.0, 0.0,
                          0.0, 1.0, 0.0],
                         size: size)
    let loss = LossFunction.crossEntropySoftmax.calculate(predicted, correct: correct)
    // Each depth: -log(p_true) / 2
    // depth 0: -log(0.7) / 2 ≈ 0.17834
    // depth 1: -log(0.6) / 2 ≈ 0.25541
    XCTAssertEqual(loss.shape, [1, 1, 2])
    XCTAssertEqual(loss.storage[0], 0.17834, accuracy: accuracy)
    XCTAssertEqual(loss.storage[1], 0.25541, accuracy: accuracy)
  }

  // MARK: - Cross Entropy Softmax with Label Smoothing

  func test_crossEntropyLoss_smoothing() {
    let predicted = Tensor([0.7, 0.1, 0.1, 0.1], size: .init(rows: 1, columns: 4, depth: 1))
    let correct = Tensor([1, 0, 0, 0], size: .init(rows: 1, columns: 4, depth: 1))

    let loss = LossFunction.crossEntropySoftmaxSmoothing(0.1).calculate(predicted, correct: correct)
    XCTAssertEqual(loss.asScalar(), 0.5026, accuracy: accuracy)

    let derivate = LossFunction.crossEntropySoftmaxSmoothing(0.1).derivative(predicted, correct: correct)
    XCTAssertTrue(derivate.isValueEqual(to: .init([-0.22499996, 0.075, 0.075, 0.075], size: predicted.size), accuracy: accuracy))
  }

  func test_crossEntropySoftmaxSmoothing_calculateScalar() {
    let predicted: [Tensor.Scalar] = [0.7, 0.1, 0.1, 0.1]
    let correct: [Tensor.Scalar] = [1.0, 0.0, 0.0, 0.0]
    let loss = LossFunction.crossEntropySoftmaxSmoothing(0.1).calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.5026, accuracy: accuracy)
  }

  func test_crossEntropySoftmaxSmoothing_zeroSmoothing_matchesCrossEntropySoftmax() {
    // With smoothing=0, should produce identical result to crossEntropySoftmax
    let predicted: [Tensor.Scalar] = [0.1, 0.7, 0.2]
    let correct: [Tensor.Scalar] = [0.0, 1.0, 0.0]
    let smoothedLoss = LossFunction.crossEntropySoftmaxSmoothing(0.0).calculate(predicted, correct: correct)
    let ceLoss = LossFunction.crossEntropySoftmax.calculate(predicted, correct: correct)
    XCTAssertEqual(smoothedLoss, ceLoss, accuracy: accuracy)
  }

  // MARK: - Binary Cross Entropy

  func test_binaryCrossEntropy_calculateScalar() {
    // i=0: -log(0.8) ≈ 0.22314; i=1: -log(0.8) ≈ 0.22314; total ≈ 0.44629
    let predicted: [Tensor.Scalar] = [0.8, 0.2]
    let correct: [Tensor.Scalar] = [1.0, 0.0]
    let loss = LossFunction.binaryCrossEntropy.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.44629, accuracy: accuracy)
  }

  func test_binaryCrossEntropy_calculateTensor() {
    let size = TensorSize(rows: 1, columns: 2, depth: 1)
    let predicted = Tensor([0.8, 0.2], size: size)
    let correct = Tensor([1.0, 0.0], size: size)
    let loss = LossFunction.binaryCrossEntropy.calculate(predicted, correct: correct)
    XCTAssertEqual(loss.asScalar(), 0.44629, accuracy: accuracy)
  }

  func test_binaryCrossEntropy_derivative() {
    // -1 * ((y/p) - (1-y)/(1-p))
    // = -1 * ([1.25, 0] - [0, 1.25]) = [-1.25, 1.25]
    let size = TensorSize(rows: 1, columns: 2, depth: 1)
    let predicted = Tensor([0.8, 0.2], size: size)
    let correct = Tensor([1.0, 0.0], size: size)
    let derivative = LossFunction.binaryCrossEntropy.derivative(predicted, correct: correct)
    let expected = Tensor([-1.25, 1.25], size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: accuracy))
  }

  func test_binaryCrossEntropy_perfectPrediction() {
    // Both predictions perfectly match labels → near-zero loss
    let predicted: [Tensor.Scalar] = [0.9999, 0.0001]
    let correct: [Tensor.Scalar] = [1.0, 0.0]
    let loss = LossFunction.binaryCrossEntropy.calculate(predicted, correct: correct)
    XCTAssertLessThan(loss, 0.002)
  }

  // MARK: - Binary Cross Entropy Softmax

  func test_binaryCrossEntropySoftmax_calculateScalar() {
    // Same forward formula as binaryCrossEntropy
    let predicted: [Tensor.Scalar] = [0.8, 0.2]
    let correct: [Tensor.Scalar] = [1.0, 0.0]
    let loss = LossFunction.binaryCrossEntropySoftmax.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.44629, accuracy: accuracy)
  }

  func test_binaryCrossEntropySoftmax_derivative() {
    // Uses crossEntropySoftmax derivative: predicted - correct
    let size = TensorSize(rows: 1, columns: 2, depth: 1)
    let predicted = Tensor([0.8, 0.2], size: size)
    let correct = Tensor([1.0, 0.0], size: size)
    let derivative = LossFunction.binaryCrossEntropySoftmax.derivative(predicted, correct: correct)
    let expected = Tensor([-0.2, 0.2], size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: accuracy))
  }

  // MARK: - Wasserstein

  func test_wasserstein_calculateScalar() {
    let predicted: [Tensor.Scalar] = [0.7]
    let correct: [Tensor.Scalar] = [1.0]
    let loss = LossFunction.wasserstein.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.7, accuracy: accuracy)
  }

  func test_wasserstein_calculateTensor() {
    let size = TensorSize(rows: 1, columns: 1, depth: 1)
    let predicted = Tensor([0.7], size: size)
    let correct = Tensor([1.0], size: size)
    let loss = LossFunction.wasserstein.calculate(predicted, correct: correct)
    XCTAssertEqual(loss.asScalar(), 0.7, accuracy: accuracy)
  }

  func test_wasserstein_derivative_returnsCorrect() {
    let size = TensorSize(rows: 1, columns: 1, depth: 1)
    let predicted = Tensor([0.7], size: size)
    let correct = Tensor([1.0], size: size)
    let derivative = LossFunction.wasserstein.derivative(predicted, correct: correct)
    XCTAssertTrue(derivative.isValueEqual(to: correct, accuracy: accuracy))
  }

  func test_wasserstein_negativeLabel() {
    // Wasserstein typically uses -1/+1 labels for critic
    let predicted: [Tensor.Scalar] = [0.8]
    let correct: [Tensor.Scalar] = [-1.0]
    let loss = LossFunction.wasserstein.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, -0.8, accuracy: accuracy)
  }

  func test_wasserstein_invalidCountReturnsZero() {
    // More than 1 element → returns 0
    let predicted: [Tensor.Scalar] = [0.7, 0.3]
    let correct: [Tensor.Scalar] = [1.0, 0.0]
    let loss = LossFunction.wasserstein.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.0, accuracy: accuracy)
  }

  // MARK: - Minimax Binary Cross Entropy

  func test_minimaxBinaryCrossEntropy_calculateScalar() {
    // Same forward formula as binaryCrossEntropy
    let predicted: [Tensor.Scalar] = [0.8, 0.2]
    let correct: [Tensor.Scalar] = [1.0, 0.0]
    let loss = LossFunction.minimaxBinaryCrossEntropy.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.44629, accuracy: accuracy)
  }

  func test_minimaxBinaryCrossEntropy_derivative() {
    // Same derivative formula as binaryCrossEntropy
    let size = TensorSize(rows: 1, columns: 2, depth: 1)
    let predicted = Tensor([0.8, 0.2], size: size)
    let correct = Tensor([1.0, 0.0], size: size)
    let derivative = LossFunction.minimaxBinaryCrossEntropy.derivative(predicted, correct: correct)
    let expected = Tensor([-1.25, 1.25], size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: accuracy))
  }

  // MARK: - Focal Softmax

  func test_focalSoftmax_calculateScalar() {
    // alpha=0.25, gamma=2, p=0.7
    // -0.25 * (0.3)^2 * log(0.7) = -0.25 * 0.09 * (-0.35667) ≈ 0.008025
    let predicted: [Tensor.Scalar] = [0.1, 0.7, 0.2]
    let correct: [Tensor.Scalar] = [0.0, 1.0, 0.0]
    let loss = LossFunction.focalSoftmax(alpha: 0.25, gamma: 2.0).calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.008025, accuracy: accuracy)
  }

  func test_focalSoftmax_calculateTensor() {
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.1, 0.7, 0.2], size: size)
    let correct = Tensor([0.0, 1.0, 0.0], size: size)
    let loss = LossFunction.focalSoftmax(alpha: 0.25, gamma: 2.0).calculate(predicted, correct: correct)
    XCTAssertEqual(loss.asScalar(), 0.008025, accuracy: accuracy)
  }
  
  func test_focalSoftmax_derivative_hardExample() {
    // Hard example: true class has very low probability (p_t = 0.01)
    // This is the regime where focal loss should produce meaningful gradients
    // and where the old buggy derivative would explode.
    //
    // pt=0.01, α=0.25, γ=2.0
    // G = α · (1-p_t)^(γ-1) · [(1-p_t) - γ·p_t·log(p_t)]
    //   = 0.25 · 0.99^1 · (0.99 - 2·0.01·log(0.01))
    //   = 0.25 · 0.99 · (0.99 - 2·0.01·(-4.60517))
    //   = 0.25 · 0.99 · (0.99 + 0.092103)
    //   = 0.25 · 0.99 · 1.082103
    //   ≈ 0.267821
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.495, 0.01, 0.495], size: size)
    let correct = Tensor([0.0, 1.0, 0.0], size: size)
    let derivative = LossFunction.focalSoftmax(alpha: 0.25, gamma: 2.0).derivative(predicted, correct: correct)
    // result_j = G · (p_j - y_j)
    // j=0: 0.267821 ·  0.495  ≈  0.132571
    // j=1: 0.267821 · -0.990  ≈ -0.265143
    // j=2: 0.267821 ·  0.495  ≈  0.132571
    let expected = Tensor([0.132571, -0.265143, 0.132571], size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: 0.001))
    
    // Sanity check: gradient for true class should be bounded (not explode)
    // With the old buggy derivative, this value would have been ~-2.01 (7.6× too large)
    let trueClassGrad = derivative.storage[1]
    XCTAssertTrue(abs(trueClassGrad) < 1.0,
                  "True-class gradient should be bounded; got \(trueClassGrad)")
  }

  func test_focalSoftmax_derivative() {
    // pt=0.7, G = 0.25 * 0.3 * (0.3 - 2*0.7*log(0.7)) ≈ 0.059951
    // result_j = G * (p_j - y_j)
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.1, 0.7, 0.2], size: size)
    let correct = Tensor([0.0, 1.0, 0.0], size: size)
    let derivative = LossFunction.focalSoftmax(alpha: 0.25, gamma: 2.0).derivative(predicted, correct: correct)
    // j=0: 0.059951 *  0.1 ≈  0.005995
    // j=1: 0.059951 * -0.3 ≈ -0.017985
    // j=2: 0.059951 *  0.2 ≈  0.011990
    let expected = Tensor([0.005995, -0.017985, 0.011990], size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: 0.001))
  }

  func test_focalSoftmax_highConfidence_reducesLoss() {
    // With gamma > 0, focal loss down-weights easy (high confidence) examples
    let predicted: [Tensor.Scalar] = [0.05, 0.9, 0.05]
    let correct: [Tensor.Scalar] = [0.0, 1.0, 0.0]
    let focalLoss = LossFunction.focalSoftmax(alpha: 1.0, gamma: 2.0).calculate(predicted, correct: correct)
    let ceLoss = LossFunction.crossEntropy.calculate(predicted, correct: correct)
    // Focal loss should be less than CE for well-classified samples
    XCTAssertLessThan(focalLoss, ceLoss)
  }

  // MARK: - Huber

  func test_huber_calculateScalar() {
    // delta=1.0, predicted=[0.5, 2.0, 0.2], correct=[1.0, 0.0, 0.0]
    // e[0]=0.5:  |e|≤δ → 0.5 * 0.5^2 = 0.125
    // e[1]=-2.0: |e|>δ → 1.0 * (2.0 - 0.5) = 1.5
    // e[2]=-0.2: |e|≤δ → 0.5 * 0.04 = 0.02
    // mean = (0.125 + 1.5 + 0.02) / 3 ≈ 0.54833
    let predicted: [Tensor.Scalar] = [0.5, 2.0, 0.2]
    let correct: [Tensor.Scalar] = [1.0, 0.0, 0.0]
    let loss = LossFunction.huber(delta: 1.0).calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.54833, accuracy: accuracy)
  }

  func test_huber_calculateTensor() {
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.5, 2.0, 0.2], size: size)
    let correct = Tensor([1.0, 0.0, 0.0], size: size)
    let loss = LossFunction.huber(delta: 1.0).calculate(predicted, correct: correct)
    XCTAssertEqual(loss.asScalar(), 0.54833, accuracy: accuracy)
  }

  func test_huber_derivative() {
    // delta=1.0, predicted=[0.5, 2.0, 0.2], correct=[1.0, 0.0, 0.0], N=3
    // e[0]=0.5:  |e|≤δ → -e/N = -0.5/3 ≈ -0.16667
    // e[1]=-2.0: |e|>δ → -δ*sign(e)/N = -1*(-1)/3 = 1/3 ≈ 0.33333
    // e[2]=-0.2: |e|≤δ → -e/N = 0.2/3 ≈ 0.06667
    let size = TensorSize(rows: 1, columns: 3, depth: 1)
    let predicted = Tensor([0.5, 2.0, 0.2], size: size)
    let correct = Tensor([1.0, 0.0, 0.0], size: size)
    let derivative = LossFunction.huber(delta: 1.0).derivative(predicted, correct: correct)
    let expected = Tensor([-0.5 / 3.0, 1.0 / 3.0, 0.2 / 3.0] as Tensor.Value, size: size)
    XCTAssertTrue(derivative.isValueEqual(to: expected, accuracy: accuracy))
  }

  func test_huber_perfectPrediction() {
    let predicted: [Tensor.Scalar] = [1.0, 0.0, 0.0]
    let correct: [Tensor.Scalar] = [1.0, 0.0, 0.0]
    let loss = LossFunction.huber(delta: 1.0).calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.0, accuracy: accuracy)
  }

  // MARK: - Edge Cases

  func test_calculateScalar_mismatchedCountReturnsZero() {
    let predicted: [Tensor.Scalar] = [0.5, 0.3]
    let correct: [Tensor.Scalar] = [1.0, 0.0, 0.0]
    let loss = LossFunction.meanSquareError.calculate(predicted, correct: correct)
    XCTAssertEqual(loss, 0.0, accuracy: accuracy)
  }
}

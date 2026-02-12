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
  
  func test_crossEntropyLoss_smoothing() {
    let predicted = Tensor([0.7, 0.1, 0.1, 0.1], size: .init(rows: 1, columns: 4, depth: 1))
    
    let correct = Tensor([1, 0, 0, 0], size: .init(rows: 1, columns: 4, depth: 1))

    let loss = LossFunction.crossEntropySoftmaxSmoothing(0.1).calculate(predicted, correct: correct)
    
    XCTAssertEqual(loss.asScalar(), 0.5026, accuracy: 0.0001)
    
    let derivate = LossFunction.crossEntropySoftmaxSmoothing(0.1).derivative(predicted, correct: correct)
    
    XCTAssertTrue(derivate.isValueEqual(to: .init([-0.22499996, 0.075, 0.075, 0.075], size: predicted.size), accuracy: 0.0001))
  }
}

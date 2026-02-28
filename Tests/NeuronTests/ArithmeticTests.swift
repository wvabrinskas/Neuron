//
//  ArithmeticTests.swift
//  Neuron
//
//  Created by William Vabrinskas on 2/27/26.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

final class ArithmeticTests: XCTestCase {
  
  func test_multiplyGradient() {
    let x = Tensor(2)
    let y = Tensor(3)
    
    let f = x * y
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), -24)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), -16)
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(-16))
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(-24))
  }
  
  func test_multiply_inverseGradient() {
    let x = Tensor(2)
    let y = Tensor(3)
    
    let f = y * x
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), -24)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), -16)
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(-16))
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(-24))
  }
  
  func test_addGradient() {
    let x = Tensor(2)
    let y = Tensor(3)
    
    let f = x + y
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), -10)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), -10)
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains([-10, -10]))
  }
  
  func test_add_inverseGradient() {
    let x = Tensor(2)
    let y = Tensor(3)
    
    let f = y + x
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), -10)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), -10)
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains([-10, -10]))
  }
  
  func test_subGradient() {
    let x = Tensor(2)
    let y = Tensor(3)
    
    let f = x - y
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), -22)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), 22)
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(22))
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(-22))
  }
  
  func test_sub_inverseGradient() {
    let x = Tensor(2)
    let y = Tensor(3)
    
    let f = y - x
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), 18)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), -18)
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(18))
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(-18))
  }
  
  func test_divGradient() {
    let x = Tensor(2)
    let y = Tensor(3)
    
    let f = x / y
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), -6.2222223, accuracy: 0.00001)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), 4.148148, accuracy: 0.00001)
  }
  
  func test_div_inverseGradient() {
    let x = Tensor(2)
    let y = Tensor(3)
    
    let f = y / x
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), 12.75, accuracy: 0.00001)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), -8.5, accuracy: 0.00001)
  }
  
  
  func test_addLayer() {
    let dense = Dense(1,
                      inputs: 1,
                      linkId: "shortcut")
    
    let sequential = Sequential (
      dense,
      ReLu(),
      Add(linkTo: "shortcut")
    )
    
    sequential.compile()
    
    dense.weights = .init(1.0)
    
    let input = Tensor(2)
    
    let out = sequential(input, context: .init())
    
    let loss = LossFunction.meanSquareError.derivative(out, correct: .init(10.0))
    
    let gradients = out.gradients(delta: loss, wrt: input)
    
    XCTAssertEqual(gradients.input.first!.asScalar(), -24)
    XCTAssertEqual(gradients.input[safe: 1, Tensor()].asScalar(), -12)

    XCTAssertEqual(gradients.input.count, sequential.layers.count)
    
    XCTAssertEqual(out.asScalar(), 4.0)
  }
  
  func test_multLayer() {
    let dense = Dense(1,
                      inputs: 1,
                      linkId: "shortcut")
    
    let sequential = Sequential (
      dense,
      ReLu(),
      Multiply(linkTo: "shortcut")
    )
    
    sequential.compile()
    
    dense.weights = .init(1.0)
    
    let input = Tensor(2)
    
    let out = sequential(input, context: .init())
    
    let loss = LossFunction.meanSquareError.derivative(out, correct: .init(10.0))
    
    let gradients = out.gradients(delta: loss, wrt: input)
    
    XCTAssertEqual(gradients.input.first!.asScalar(), -48)
    XCTAssertEqual(gradients.input[safe: 1, Tensor()].asScalar(), -24)

    XCTAssertEqual(gradients.input.count, sequential.layers.count)
    
    XCTAssertEqual(out.asScalar(), 4.0)
  }
  
  func test_subtractLayer() {
    let dense = Dense(1,
                      inputs: 1,
                      linkId: "shortcut")
    
    let sequential = Sequential (
      dense,
      ReLu(),
      Subtract(linkTo: "shortcut")
    )
    
    sequential.compile()
    
    dense.weights = .init(1.0)
    
    let input = Tensor(2)
    
    let out = sequential(input, context: .init())
    
    let loss = LossFunction.meanSquareError.derivative(out, correct: .init(10.0))
    
    let gradients = out.gradients(delta: loss, wrt: input)
    
    XCTAssertEqual(gradients.input.first!.asScalar(), 40)
    XCTAssertEqual(gradients.input[safe: 1, Tensor()].asScalar(), 20)

    XCTAssertEqual(gradients.input.count, sequential.layers.count)
    
    XCTAssertEqual(out.asScalar(), 0)
  }
  
//  func test_subtract_inverseLayer() {
//    let dense = Dense(1,
//                      inputs: 1,
//                      linkId: "shortcut")
//    
//    let sequential = Sequential (
//      dense,
//      ReLu(),
//      Subtract(inverse: true,
//               linkTo: "shortcut")
//    )
//    
//    sequential.compile()
//    
//    dense.weights = .init(1.0)
//    
//    let input = Tensor(2)
//    
//    let out = sequential(input, context: .init())
//    
//    let loss = LossFunction.meanSquareError.derivative(out, correct: .init(10.0))
//    
//    let gradients = out.gradients(delta: loss, wrt: input)
//    
//    XCTAssertEqual(gradients.input.first!.asScalar(), 60)
//    XCTAssertEqual(gradients.input[safe: 1, Tensor()].asScalar(), 20)
//
//    XCTAssertEqual(gradients.input.count, sequential.layers.count)
//    
//    XCTAssertEqual(out.asScalar(), 0)
//  }
}

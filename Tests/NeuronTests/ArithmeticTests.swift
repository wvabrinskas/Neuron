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
 
  func test_multiMathGradient_withDiv() {
    let x = Tensor(2)
    let y = Tensor(8)
    let z = Tensor(4)
    let w = Tensor(6)
    
    x.label = "x"
    y.label = "y"
    z.label = "z"
    w.label = "w"

    let f = x * y + z / w
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), 106.66666, accuracy: 1e-6)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), 26.666664, accuracy: 1e-6)
    XCTAssertEqual(f.gradients(delta: loss, wrt: z).input.first!.asScalar(), 2.222222, accuracy: 1e-6)
    XCTAssertEqual(f.gradients(delta: loss, wrt: w).input.first!.asScalar(), -1.4814813, accuracy: 1e-6)
  }
  
  func test_multiMathGradient_withSub() {
    let x = Tensor(2)
    let y = Tensor(8)
    let z = Tensor(4)
    let w = Tensor(6)
    
    x.label = "x"
    y.label = "y"
    z.label = "z"
    w.label = "w"

    let f = x * y + z - w
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), 64, accuracy: 1e-6)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), 16, accuracy: 1e-6)
    XCTAssertEqual(f.gradients(delta: loss, wrt: z).input.first!.asScalar(), 8, accuracy: 1e-6)
    XCTAssertEqual(f.gradients(delta: loss, wrt: w).input.first!.asScalar(), -8, accuracy: 1e-6)
  }
  
  func test_multiMathGradient() {
    let x = Tensor(2)
    let y = Tensor(8)
    let z = Tensor(4)
    
    let f = x * y + z
    
    let loss = LossFunction.meanSquareError.derivative(f, correct: .init(10.0))
   
    XCTAssertEqual(f.gradients(delta: loss, wrt: x).input.first!.asScalar(), 160)
    XCTAssertEqual(f.gradients(delta: loss, wrt: y).input.first!.asScalar(), 40)
    XCTAssertEqual(f.gradients(delta: loss, wrt: z).input.first!.asScalar(), 20)
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(160))
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(40))
    XCTAssertTrue(f.gradients(delta: loss).input.map { $0.asScalar() }.contains(20))
  }
  
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
  
  func test_subtractLayer_inverse() {
    let dense = Dense(1,
                      inputs: 1,
                      linkId: "shortcut")
    
    let sequential = Sequential (
      dense,
      ReLu(),
      Subtract(inverse: false, linkTo: "shortcut")
    )
    
    sequential.compile()
    
    dense.weights = .init(1.0)
    
    let input = Tensor(2)
    
    let out = sequential(input, context: .init())
    
    let loss = LossFunction.meanSquareError.derivative(out, correct: .init(10.0))
    
    let gradients = out.gradients(delta: loss, wrt: input)
    
    XCTAssertEqual(gradients.input.count, sequential.layers.count)
    
    XCTAssertEqual(out.asScalar(), 0)
  }
  
  func test_divideLayer() {
    let dense = Dense(1,
                      inputs: 1,
                      linkId: "shortcut")
    
    let sequential = Sequential (
      dense,
      ReLu(),
      Divide(linkTo: "shortcut")
    )
    
    sequential.compile()
    
    dense.weights = .init(2.0)
    
    let input = Tensor(6)
    
    let out = sequential(input, context: .init())
    
    let loss = LossFunction.meanSquareError.derivative(out, correct: .init(10.0))
    
    let gradients = out.gradients(delta: loss, wrt: input)
    
    XCTAssertEqual(gradients.input.first!.asScalar(), 6.0)
    XCTAssertEqual(gradients.input[safe: 1, Tensor()].asScalar(), 1.5)

    XCTAssertEqual(gradients.input.count, sequential.layers.count)
    
    XCTAssertEqual(out.asScalar(), 1.0)
  }
  
  func test_divideLayer_inverse() {
    let dense = Dense(1,
                      inputs: 1,
                      linkId: "shortcut")
    
    let sequential = Sequential (
      dense,
      ReLu(),
      Divide(inverse: true,
             linkTo: "shortcut")
    )
    
    sequential.compile()
    
    dense.weights = .init(2.0)
    
    let input = Tensor(6)
    
    let out = sequential(input, context: .init())
    
    let loss = LossFunction.meanSquareError.derivative(out, correct: .init(10.0))
    
    let gradients = out.gradients(delta: loss, wrt: input)
    
    XCTAssertEqual(gradients.input.first!.asScalar(), 6.0)
    XCTAssertEqual(gradients.input[safe: 1, Tensor()].asScalar(), 1.5)

    XCTAssertEqual(gradients.input.count, sequential.layers.count)
    
    XCTAssertEqual(out.asScalar(), 1.0)
  }
}

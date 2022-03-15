//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/17/22.
//

import Foundation
import XCTest
import GameKit
@testable import Neuron
import Combine
import Accelerate
import NumSwift

final class ConvTests: XCTestCase {
  var cancellables: Set<AnyCancellable> = []
  let mnist = MNIST()
  
  override func setUp() {
    super.setUp()

  }
  
  private lazy var convBrain: ConvBrain = {
    let brain = ConvBrain(epochs: 30,
                          learningRate: 0.001,
                          bias: 1.0,
                          inputSize: (28,28,1),
                          batchSize: 8,
                          initializer: .heNormal)
    
    brain.addConvolution(filterCount: 16)
    brain.addMaxPool()
    brain.addConvolution(filterCount: 32)
    brain.addMaxPool()
    brain.addDenseNormal(128, rate: 0.001, momentum: 0.9)
    brain.addDenseNormal(64, rate: 0.001, momentum: 0.9)
    brain.addDense(10, activation: .softmax)
    
    brain.compile()
    
    brain.logLevel = .low
    
    return brain
  }()

//  func testConvLobe() async {
//    let dataset = await mnist.build()
//    convBrain.train(data: dataset)
//  }
  
  func testPooling() {
    let inputTensor = (10,10,3)
    let outputShape = [5,5,3]
    
    let inputData = NumSwift.zerosLike(inputTensor)
    let lobe = PoolingLobe(model: .init(inputSize: inputTensor))
    
    let output = lobe.feed(inputs: inputData, training: true)
    let testOutputTensor = output.shape
    
    XCTAssertEqual(testOutputTensor, outputShape, "Pooling lobe output shape is broken")
    
    let backwardOutput = lobe.calculateGradients(with: output)
    XCTAssertEqual(backwardOutput.shape, [inputTensor.0, inputTensor.1, inputTensor.2], "Pooling lobe backward shape is broken")
  }
  
  func testConvolution() {
    let inputTensor = (10,10,3)
    
    let filterCount = 6
    let outputShape = [10,10,filterCount]
    
    let input: [[[Float]]] = NumSwift.zerosLike(inputTensor)
    
    let lobe = ConvolutionalLobe(model: .init(inputSize: inputTensor,
                                              activation: .reLu,
                                              bias: 0,
                                              filterSize: (3,3,3),
                                              filterCount: filterCount),
                                 learningRate: 0.001,
                                 initializer: .init(type: .heNormal))
    
    let output = lobe.feed(inputs: input, training: true)
    XCTAssertEqual(output.shape, outputShape, "Convolution lobe output shape is broken")
    
    let backwardOutput = lobe.calculateGradients(with: output)
    XCTAssertEqual(backwardOutput.shape, [inputTensor.0, inputTensor.1, inputTensor.2], "Convolution lobe backward shape is broken")
  }
  
  func testFlatten() {
    let inputTensor = (10,10,3)
    let input: [[[Float]]] = NumSwift.zerosLike(inputTensor)
    
    let expected = inputTensor.0 * inputTensor.1 * inputTensor.2
    
    let lobe = Flatten()
    let output = lobe.feed(inputs: input, training: true)
    
    XCTAssertEqual(output.count, expected, "Flatten is broken")
    
    let backwardOutput = lobe.backpropagate(deltas: output)
    XCTAssertEqual(backwardOutput.shape, [inputTensor.0, inputTensor.1, inputTensor.2], "Convolution lobe backward shape is broken")
  }
  
  func testFullyConnectedExists() {
    let brain = convBrain
    XCTAssertTrue(brain.fullyConnected.compiled)
  }

  func print3d(array: [[[Any]]]) {
    var i = 0
    array.forEach { first in
      print("index: ", i)
      first.forEach { print($0) }
      i += 1
    }
  }
}

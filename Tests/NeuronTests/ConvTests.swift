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
    brain.addDense(64, activation: .reLu)
    brain.addDense(20, activation: .reLu)
    brain.addDense(10, activation: .softmax)
    
    brain.compile()
    
    brain.logLevel = .low
    
    return brain
  }()

//  func testConvLobe() async {
//    let dataset = await mnist.build()
//    convBrain.train(data: dataset)
//  }

  func print3d(array: [[[Any]]]) {
    var i = 0
    array.forEach { first in
      print("index: ", i)
      first.forEach { print($0) }
      i += 1
    }
  }
}

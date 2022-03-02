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
                          inputSize: (28,28,1),
                          batchSize: 64)
    
    brain.addConvolution(filterCount: 6)
    brain.addMaxPool()
    brain.addConvolution(filterCount: 16)
    brain.addMaxPool()
    brain.addDense(64)
    brain.addDense(10, activation: .none)
    brain.addSoftmax()
    
    brain.compile()
    
    return brain
  }()

  func testConvLobe() {
    var dataset: DatasetData?
    
    let expectation = XCTestExpectation()
    
    mnist.dataPublisher
      .receive(on: DispatchQueue.main)
      .sink { val in
        dataset = val
        expectation.fulfill()
      }
      .store(in: &self.cancellables)
    
    mnist.build()
    
    wait(for: [expectation], timeout: 1000)
    
    guard let dataset = dataset else {
      XCTFail()
      return
    }

    convBrain.train(data: dataset) { epoch in
      print(self.convBrain.loss.last)
    } completed: { loss in
      //print(loss)
    }
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
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

final class ConvTests: XCTestCase {
  var cancellables: Set<AnyCancellable> = []

  private lazy var imageGray: [[[Float]]] = {
    let r: [[Float]] = [[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 1, 1, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]
    
    let g: [[Float]] = [[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]
    
    let b: [[Float]] = [[0, 0, 0, 0, 0, 0, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0.5, 0, 0, 0.5, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 0]]
    return [r, g, b]
  }()
  
  private lazy var image: [Float] = {
    let img: [[Float]] = [[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]]
    
    return img.flatMap { $0 }
  }()
  
  private lazy var image2: [Float] = {
    let img: [[Float]] = [[0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0],
                          [0.5, 1, 1, 1, 1, 0.5, 0, 0],
                          [0.5, 1, 0, 0, 1, 0.5, 0, 0],
                          [0.5, 1, 0, 0, 1, 0.5, 0, 0],
                          [0.5, 1, 0, 0, 1, 0.5, 0, 0],
                          [0.5, 1, 0, 0, 1, 0.5, 0, 0],
                          [0.5, 1, 1, 1, 1, 0.5, 0, 0],
                          [0, 0.5, 0.5, 0.5, 0.5, 0, 0, 0]]
    
    return img.flatMap { $0 }
  }()

  private lazy var brain: Brain = {

    let b = Brain(lossFunction: .crossEntropy,
                  initializer: .xavierNormal)

    b.add(LobeModel(nodes: 256, activation: .reLu, bias: 0.001))
    b.add(LobeModel(nodes: 30, activation: .reLu, bias: 0.001))
    b.add(LobeModel(nodes: 3, activation: .reLu))

    b.add(modifier: .softmax)
    b.compile()

    return b
  }()
  
  private lazy var convBrain: ConvBrain = {
    let brain = ConvBrain(epochs: 100,
                          learningRate: 0.0001,
                          inputSize: (8,8,3),
                          batchSize: 1,
                          fullyConnected: brain)
    
    brain.addConvolution(filterCount: 32)
    brain.addMaxPool()
    brain.addConvolution(filterCount: 64)
    brain.addMaxPool()
    
    return brain
  }()

  func testConvLobe() {
    let data = ConvTrainingData(data: imageGray, label: [0,1,0])
    let brainOut = convBrain.feed(data: data)
    
    let loss = brain.getOutputDeltas(outputs: brainOut, correctValues: [0,1,0])
    
    print(brainOut)
  }
  
  func testImportMNIST() {
    let mnist = MNIST()
    
    let expectation = XCTestExpectation()
        
    mnist.trainingData.publisher.eraseToAnyPublisher()
      .sink(receiveValue: { data in
      print(data.data.count)
      expectation.fulfill()
    })
    .store(in: &cancellables)
        
    mnist.build()
    
    wait(for: [expectation], timeout: 40)
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

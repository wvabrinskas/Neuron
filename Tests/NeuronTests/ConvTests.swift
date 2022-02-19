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

final class ConvTests: XCTestCase {
  
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
    
    let b = Brain(learningRate: 0.00001,
                  epochs: 100,
                  lossFunction: .crossEntropy)
    
    b.add(LobeModel(nodes: 64))
    
    b.add(ConvolutionalLobeModel(inputSize: (8, 8),
                                 activation: .none,
                                 bias: 0))
    
    b.add(PoolingLobeModel(inputSize: (8, 8)))
    
//    b.add(ConvolutionalLobeModel(inputSize: (4, 4),
//                                 activation: .none,
//                                 bias: 0))
//
//    b.add(PoolingLobeModel(inputSize: (4, 4)))
    
    b.add(LobeModel(nodes: 10, activation: .reLu, bias: 0))
    b.add(LobeModel(nodes: 5, activation: .sigmoid, bias: 0))
    
    b.add(modifier: .softmax)
    b.add(optimizer: .adam())
    b.compile()
    
    return b
  }()
  
  func testFeed() {
    let correct: [Float] = [0,0,1,0,0]
    let correct2: [Float] = [1,0,0,0,0]
    
    let trainingData = TrainingData(data: image, correct: correct)
    let trainingData2 = TrainingData(data: image2, correct: correct2)

    for i in 0..<1000 {
      brain.zeroGradients()
      
      var image = trainingData.data
      var expected = trainingData.correct
      
      if i % 2 == 0 {
        image = trainingData2.data
        expected = trainingData2.correct
      }
      
      let result = brain.feed(input: image)
      let out = brain.getOutputDeltas(outputs: result, correctValues: expected)
      let loss = brain.loss(result, correct: expected)
      
      print(loss)
      brain.backpropagate(with: out)
      brain.adjustWeights(batchSize: 1)
      
    }

  }
}

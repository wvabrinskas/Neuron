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
    let img: [[Float]] = [[0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0],
                          [0, 0, 0, 1, 1, 0, 0, 0]]
    
    return img.flatMap { $0 }
  }()

  private lazy var brain: Brain = {
    
    let b = Brain(learningRate: 0.0001,
                  epochs: 100,
                  lossFunction: .crossEntropy)
    
    b.add(LobeModel(nodes: 64))
    
    b.add(ConvolutionalLobeModel(inputSize: (8, 8),
                                 activation: .reLu,
                                 bias: 0))
    
    b.add(PoolingLobeModel(inputSize: (8, 8)))
    
    b.add(ConvolutionalLobeModel(inputSize: (4, 4),
                                 activation: .reLu,
                                 bias: 0))
    
    b.add(PoolingLobeModel(inputSize: (4, 4)))
    
    b.add(LobeModel(nodes: 5, activation: .sigmoid, bias: 0))
    
   // b.add(modifier: .softmax)
    b.compile()
    
    return b
  }()
  
  func testFeed() {
    let result = brain.feed(input: image)
    print(result)
    
    //brain.backpropagate(with: [0, 0.1, 0.5, -0.1, -0.5])
  }
}

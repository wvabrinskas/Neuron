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
  private lazy var imageGray: [[[Float]]] = {
    let img: [[Float]] = [[0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0.5, 1, 1, 0.5, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0]]
    
    return [img]
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

    let b = Brain(learningRate: 0.00001,
                  epochs: 100,
                  lossFunction: .crossEntropy)

    b.add(LobeModel(nodes: 48, activation: .reLu, bias: 0))
    b.add(LobeModel(nodes: 10, activation: .reLu, bias: 0))
    b.add(LobeModel(nodes: 3, activation: .none, bias: 0))

    b.add(modifier: .softmax)
    b.compile()

    return b
  }()
  
  private lazy var convBrain: ConvLobe = {
    ConvLobe(model: .init(inputSize: (8,8,1),
                          activation: .reLu,
                          bias: 0,
                          filterSize: (3,3),
                          filterCount: 3),
             learningRate: 0.00001)
  }()
  
//  func testFeed() {
//    let correct: [Float] = [0,0,1,0,0]
//    let correct2: [Float] = [1,0,0,0,0]
//
//    let trainingData = TrainingData(data: image, correct: correct)
//    let trainingData2 = TrainingData(data: image2, correct: correct2)
//
//    for i in 0..<1000 {
//      brain.zeroGradients()
//
//      var image = trainingData.data
//      var expected = trainingData.correct
//
//      if i % 2 == 0 {
//        image = trainingData2.data
//        expected = trainingData2.correct
//      }
//
//      let result = brain.feed(input: image)
//      let out = brain.getOutputDeltas(outputs: result, correctValues: expected)
//      let loss = brain.loss(result, correct: expected)
//
//      print(loss)
//      brain.backpropagate(with: out)
//      brain.adjustWeights(batchSize: 1)
//
//    }

 // }
  
  func testConvLobe() {
    //conv
    let convOut = convBrain.feed(inputs: imageGray, training: true)
    
    let pool = PoolingLobe(model: .init())
    
    //pool
    let out = pool.feed(inputs: convOut, training: true)
    
    //flatten
    let flat = out.flatMap { $0.flatMap { $0 } }
    
    //fully connected
    let brainOut = brain.feed(input: flat)
    let loss = brain.getOutputDeltas(outputs: brainOut, correctValues: [0,1,0])
    
    let brainBackprop = brain.backpropagate(with: loss)

    let deltas = brainBackprop.firstLayerDeltas
    
    //get output dimensions from ConvLobe / PoolingLobe
    let outputSize = (4,4,3)
    let batchedDeltas = deltas.batched(into: outputSize.0 * outputSize.1)
   
    let gradients = batchedDeltas.map { $0.reshape(columns: outputSize.1) }
    
    let poolGradients = pool.calculateGradients(with: gradients)
    
    let convGradients = convBrain.calculateGradients(with: poolGradients)
    
    //flatten layer is just the multi dimensional output flattened to a 1D array
    //to get back to the multidimensional just reshap to multi dimensions with previous conv / pool layer
    //dimensions
    var i = 0
    convGradients.forEach { first in
      print("index: ", i)
      first.forEach { print($0) }
      i += 1
    }
  }
}

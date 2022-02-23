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

    let b = Brain(learningRate: 0.00001,
                  epochs: 100,
                  lossFunction: .crossEntropy)

    b.add(LobeModel(nodes: 512, activation: .reLu, bias: 0.001))
    b.add(LobeModel(nodes: 30, activation: .reLu, bias: 0.001))
    b.add(LobeModel(nodes: 3, activation: .reLu))

    b.add(modifier: .softmax)
    b.compile()

    return b
  }()
  
  private lazy var convBrain: ConvolutionalLobe = {
    ConvolutionalLobe(model: .init(inputSize: (8,8,3),
                                   activation: .reLu,
                                   bias: 1,
                                   filterSize: (3,3,3),
                                   filterCount: 32),
                      learningRate: 0.00001)
  }()
  
  private let flatten = Flatten()
  private let pool = PoolingLobe(model: .init())

  func testConvLobe() {
    //conv
    let convOut = convBrain.feed(inputs: imageGray, training: true)
    
    //pool
    let out = pool.feed(inputs: convOut, training: true)
    
    //flatten
    let flat = flatten.feed(inputs: out)
    

    //fully connected
    let brainOut = brain.feed(input: flat)
    let loss = brain.getOutputDeltas(outputs: brainOut, correctValues: [0,1,0])
    
    print(loss)
//
    let brainBackprop = brain.backpropagate(with: loss)
//
    let deltas = brainBackprop.firstLayerDeltas
//
//    //reverse flatten
    let gradients = flatten.backpropagate(deltas: deltas)
//
//    //pooling gradients
    let poolGradients = pool.calculateGradients(with: gradients)
    
//
//    //conv gradients
    let convGradients = convBrain.calculateGradients(with: poolGradients)
    
    print3d(array: convGradients)
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

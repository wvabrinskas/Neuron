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

    let b = Brain(lossFunction: .crossEntropy,
                  initializer: .xavierNormal)

    b.add(LobeModel(nodes: 512, activation: .reLu, bias: 0.001))
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
                          fullyConnected: brain)
    
    brain.addConvolution(filterCount: 32)
    brain.addMaxPool()
    return brain
  }()

  func testConvLobe() {
    let data = ConvTrainingData(data: imageGray, label: [0,1,0])
    let brainOut = convBrain.feed(data: data)
    
    let loss = brain.getOutputDeltas(outputs: brainOut, correctValues: [0,1,0])
    
    print(brainOut)
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

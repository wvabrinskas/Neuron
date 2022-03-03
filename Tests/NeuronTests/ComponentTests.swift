//
//  File.swift
//  
//
//  Created by William Vabrinskas on 2/4/22.
//

import XCTest
import NumSwift
@testable import Neuron


class ComponentTests: XCTestCase {
  
  public lazy var brain: Brain = {
    let bias: Float = 0
    
    let brain = Brain(learningRate: 1,
                      epochs: 1,
                      lossFunction: .binaryCrossEntropy,
                      lossThreshold: TestConstants.lossThreshold,
                      initializer: .xavierNormal,
                      descent: .bgd)
    
    brain.addInputs(1) //input layer no activation. It'll be ignored anyway
    
    for _ in 0..<1 {
      brain.add(LobeModel(nodes: 1, activation: .reLu,  bias: bias)) //hidden layer
    }
    
    brain.add(LobeModel(nodes: 1, activation: .reLu, bias: bias)) //output layer
    brain.logLevel = .low
    
    return brain
  }()
  
  public lazy var complexBrain: Brain = {
    let bias: Float = 0
    
    let brain = Brain(learningRate: 1,
                      epochs: 1,
                      lossFunction: .crossEntropy,
                      lossThreshold: TestConstants.lossThreshold,
                      initializer: .xavierNormal,
                      descent: .bgd)
    
    brain.addInputs(1) //input layer no activation. It'll be ignored anyway
    
    for _ in 0..<2 {
      brain.add(LobeModel(nodes: 2, activation: .reLu,  bias: bias)) //hidden layer
    }
    
    brain.add(LobeModel(nodes: 1, activation: .reLu, bias: bias)) //output layer
    brain.logLevel = .low
    
    return brain
  }()
  
  override func setUp() {
    super.setUp()
    
    brain.compile()
    
    let weights: [[[Float]]] = [[[0]], [[0.5]], [[0.5]]]
    
    brain.replaceWeights(weights: weights)
    
    complexBrain.compile()
    
    let complexWeights: [[[Float]]] = [[[0]], [[0.5], [0.5]], [[0.5, 0.5], [0.5, 0.5]], [[0.5, 0.5]]]
    
    complexBrain.replaceWeights(weights: complexWeights)
  }
  
  func testComplexBackprop() {
    let input: [Float] = [1.0]
        
    let result = complexBrain.feed(input: input)
    XCTAssert(result == [0.5])
    
    let error = result - Float(1.0)

    let backprop = complexBrain.backpropagate(with: error)

    let expectedGradients: [[[Float]]] = [[[-0.25], [-0.25]], [[-0.125, -0.125], [-0.125, -0.125]], [[-0.25, -0.25]]].reversed()
    XCTAssert(backprop.gradients == expectedGradients)
    complexBrain.adjustWeights(batchSize: 1)
    
    XCTAssert(complexBrain.weights() == [[[0.0]], [[0.75], [0.75]], [[0.625, 0.625], [0.625, 0.625]], [[0.75, 0.75]]])
  }
  
  func testSimpleBackprop() {
    let input: [Float] = [1.0]
        
    let result = brain.feed(input: input)
    XCTAssert(result == [0.25])
    
    let error = result - Float(1.0)

    let backprop = brain.backpropagate(with: error)
    XCTAssert(backprop.gradients ==  [[[-0.375]], [[-0.375]]])
    
    brain.adjustWeights(batchSize: 1)
    
    XCTAssert(brain.weights() == [[[0.0]], [[0.875]], [[0.875]]])
  }
  
//  func testBatchNormalizer() {
//    let batchNormalizer = BatchNormalizer(momentum: 0.99, learningRate: 0.01)
//    let input: [Float] = [0.5, 0.1, 0.5]
//    let output = batchNormalizer.normalize(activations: input, training: true)
//
//    XCTAssert(output == [0.7066101, -1.4132203, 0.7066101])
//
//    let backprop = batchNormalizer.backward(gradient: output)
//
//    XCTAssert(backprop == [0.005258338, -0.010517518, 0.005258338])
//  }
}

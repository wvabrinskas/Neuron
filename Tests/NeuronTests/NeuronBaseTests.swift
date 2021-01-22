//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/21/21.
//

import XCTest
@testable import Neuron


final class NeuronBaseTests: XCTestCase, BaseTestConfig {
  static var allTests = [
    ("testNeuronConnectionObjects", testNeuronConnectionObjects),
    ("testWeightNumbers", testWeightNumbers),
    ("testNumberOfLobesMatches", testNumberOfLobesMatches),
    ("testFeedIsntSame", testFeedIsntSame)
  ]
  
  public lazy var brain: Brain? = {
    let bias: Float = 0.001
    
    let brain = Brain(learningRate: 0.01,
                      epochs: 200,
                      lossFunction: .crossEntropy,
                      lossThreshold: TestConstants.lossThreshold,
                      initializer: .xavierNormal)
    
    brain.add(.init(nodes: TestConstants.inputs, bias: bias)) //input layer
    
    for _ in 0..<TestConstants.numOfHiddenLayers {
      brain.add(.init(nodes: TestConstants.hidden, activation: .reLu, bias: bias)) //hidden layer
    }
    
    brain.add(.init(nodes: TestConstants.outputs, bias: bias)) //output layer
    
    brain.add(modifier: .softmax)
    brain.logLevel = .none
    
    brain.compile()
    
    return brain
  }()
  
  func testFeedIsntSame() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    
    guard let brain = brain else {
      return
    }
    
    var previous: [Float] = [Float](repeating: 0.0, count: TestConstants.inputs)
    
    for i in 0..<10 {
      var inputs: [Float] = []
      for _ in 0..<TestConstants.inputs {
        inputs.append(Float.random(in: 0...1))
      }
      
      let out = brain.feed(input: inputs)
      
      print("Feed \(i): \(out)")
      XCTAssertTrue(previous != out, "Result is the same check code...")
      previous = out
    }
    
  }
  
  //checks to see if the neurontransmitter objects are unique
  func testNeuronConnectionObjects() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    
    guard let brain = brain else {
      return
    }
    
    brain.lobes.forEach { (lobe) in
      lobe.neurons.forEach { (neuron) in
        neuron.inputs.forEach { (connection) in
          let count = neuron.inputs.filter({ $0 == connection })
          XCTAssertTrue(count.count == 1, "Multiples of the same NeuroTransmitter")
        }
      }
    }
  }
  
  func testNumberOfLobesMatches() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    
    guard let brain = brain else {
      return
    }
    
    let inputLayer = brain.lobes.filter({ $0.layer == .input })
    let hiddenLayers = brain.lobes.filter({ $0.layer == .hidden })
    let outputLayer = brain.lobes.filter({ $0.layer == .output })

    XCTAssertTrue(inputLayer.count == 1, "Should only have 1 first layer")

    if let first = inputLayer.first {
      XCTAssertTrue(first.neurons.count == TestConstants.inputs, "Input layer count does not match model")
    }
    
    XCTAssertTrue(hiddenLayers.count == TestConstants.numOfHiddenLayers, "Number of hidden layers does not match model")
    
    hiddenLayers.forEach { (layer) in
      XCTAssertTrue(layer.neurons.count == TestConstants.hidden, "Hidden layer count does not match model")
    }
    
    XCTAssertTrue(outputLayer.count == 1, "Should only have 1 first layer")

    if let first = outputLayer.first {
      XCTAssertTrue(first.neurons.count == TestConstants.outputs, "Output layer count does not match model")
    }
    
  }
  
  func testWeightNumbers() {
    var expected = TestConstants.inputs

    for n in 0..<TestConstants.numOfHiddenLayers {
      if n == 0 {
        expected += (TestConstants.inputs * TestConstants.hidden)
      } else {
        expected += (TestConstants.hidden * TestConstants.hidden)
      }
    }
    
    expected += (TestConstants.hidden * TestConstants.outputs)
    
    let flattenedWeightsArray = flattenedWeights()
    
    XCTAssertTrue(flattenedWeightsArray.count == expected,
                  "got: \(flattenedWeightsArray.count) expected: \(expected)")
  }
  
}

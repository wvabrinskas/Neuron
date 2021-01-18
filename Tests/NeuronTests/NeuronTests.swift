import XCTest
@testable import Neuron

final class NeuronTests: XCTestCase {

  let outputs = 3
  let inputs = 4
  let hidden = 5
  
  private lazy var brain: Brain = {
    
    let nucleus = Nucleus(learningRate: 0.001,
                          bias: 0.001)
    
    let brain = Brain(nucleus: nucleus,
                      epochs: 10,
                      lossFunction: .crossEntropy,
                      lossThreshold: 0.001)
    
    brain.add(.layer(inputs, .none, .input)) //input  layer
    brain.add(.layer(hidden, .reLu, .hidden)) //hidden layer
    brain.add(.layer(outputs, Activation.none, .output)) //output layer
    
    brain.add(modifier: .softmax)
    brain.logLevel = .high
    
    return brain
  }()
  
  
  override func setUp() {
    super.setUp()
    brain.compile()
    XCTAssertTrue(brain.compiled, "Brain not initialized")
  }
  
  func testNumberOfLobesMatches() {
    
  }
  
  //checks to see if the neurontransmitter objects are unique
  func testNeuronConnectionObjects() {
    brain.lobes.forEach { (lobe) in
      lobe.neurons.forEach { (neuron) in
        neuron.inputs.forEach { (connection) in
          let count = neuron.inputs.filter({ $0 == connection })
          XCTAssertTrue(count.count == 1, "Multiples of the same NeuroTransmitter")
        }
      }
    }
  }
  
  static var allTests = [
    ("testNeuronConnectionObjects", testNeuronConnectionObjects),
  ]
}

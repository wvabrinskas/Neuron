import XCTest
@testable import Neuron

protocol Numeric {
  var asDouble: Double { get }
  init(_: Double)
}

extension Int: Numeric {var asDouble: Double { get {return Double(self)}}}
extension Float: Numeric {var asDouble: Double { get {return Double(self)}}}
extension Double: Numeric {var asDouble: Double { get {return Double(self)}}}
extension CGFloat: Numeric {var asDouble: Double { get {return Double(self)}}}

extension Array where Element: Numeric {
  
  var sd : Element { get {
    let sss = self.reduce((0.0, 0.0)){ return ($0.0 + $1.asDouble, $0.1 + ($1.asDouble * $1.asDouble))}
    let n = Double(self.count)
    return Element(sqrt(sss.1/n - (sss.0/n * sss.0/n)))
  }}
  
  
  var mean : Element { get {
    let sum = self.reduce(0.asDouble, { x, y in
      x.asDouble + y.asDouble
    })
    
    return Element(sum / Double(self.count))
  }}
}

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
  
  func testWeights() {
    var flattenedWeights: [Float] = []
    
    for layer in brain.layerWeights {
      layer.forEach { (float) in
        flattenedWeights.append(float)
      }
    }
    
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

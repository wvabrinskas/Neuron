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

final class NeuronClassificationTests: XCTestCase {

  let inputs = 4 //UIColor values rgba
  let hidden = 5
  let outputs = ColorType.allCases.count
  let numOfHiddenLayers = 1
  
  let lossThreshold: Float = 0.001
  let testingLossThreshold: Float = 0.2 //if below 0.2 considered trained
  
  static var allTests = [
    ("testNeuronConnectionObjects", testNeuronConnectionObjects),
    ("testWeightNumbers", testWeightNumbers),
    ("testNumberOfLobesMatches", testNumberOfLobesMatches),
    ("testFeedIsntSame", testFeedIsntSame),
    ("testTraining", testTraining)
  ]
  
  private lazy var brain: Brain = {
    
    let nucleus = Nucleus(learningRate: 0.01,
                          bias: 0.001)
    
    let brain = Brain(nucleus: nucleus,
                      epochs: 200,
                      lossFunction: .crossEntropy,
                      lossThreshold: lossThreshold,
                      initializer: .xavierNormal)
    
    brain.add(.layer(inputs, .none, .input)) //input  layer
    
    for _ in 0..<numOfHiddenLayers {
      brain.add(.layer(hidden, .reLu, .hidden)) //hidden layer
    }
    
    brain.add(.layer(outputs, Activation.none, .output)) //output layer
    
    brain.add(modifier: .softmax)
    brain.logLevel = .none
    
    return brain
  }()
  
  public var trainingData: [TrainingData] = []
  public var validationData: [TrainingData] = []
  
  override func setUp() {
    super.setUp()
    brain.compile()
    XCTAssertTrue(brain.compiled, "Brain not initialized")
    
    self.buildTrainingData()
  }
  
  
  func buildTrainingData() {
    let num = 100

    for _ in 0..<num {
      trainingData.append(TrainingData(data: ColorType.red.color(), correct: ColorType.red.correctValues()))
      validationData.append(TrainingData(data: ColorType.red.color(), correct: ColorType.red.correctValues()))
    }
    
    for _ in 0..<num {
      trainingData.append(TrainingData(data: ColorType.green.color(), correct: ColorType.green.correctValues()))
      validationData.append(TrainingData(data: ColorType.green.color(), correct: ColorType.green.correctValues()))
    }
    
    for _ in 0..<num {
      trainingData.append(TrainingData(data: ColorType.blue.color(), correct: ColorType.blue.correctValues()))
      validationData.append(TrainingData(data: ColorType.blue.color(), correct: ColorType.blue.correctValues()))
    }
    
  }
  
  func testNumberOfLobesMatches() {
    let inputLayer = self.brain.lobes.filter({ $0.layer == .input })
    let hiddenLayers = self.brain.lobes.filter({ $0.layer == .hidden })
    let outputLayer = self.brain.lobes.filter({ $0.layer == .output })

    XCTAssertTrue(inputLayer.count == 1, "Should only have 1 first layer")

    if let first = inputLayer.first {
      XCTAssertTrue(first.neurons.count == inputs, "Input layer count does not match model")
    }
    
    XCTAssertTrue(hiddenLayers.count == numOfHiddenLayers, "Number of hidden layers does not match model")
    
    hiddenLayers.forEach { (layer) in
      XCTAssertTrue(layer.neurons.count == hidden, "Hidden layer count does not match model")
    }
    
    XCTAssertTrue(outputLayer.count == 1, "Should only have 1 first layer")

    if let first = outputLayer.first {
      XCTAssertTrue(first.neurons.count == outputs, "Output layer count does not match model")
    }
    
  }
  
  func testWeightNumbers() {
    var flattenedWeights: [Float] = []
    
    for layer in brain.layerWeights {
      layer.forEach { (float) in
        flattenedWeights.append(float)
      }
    }
    
    let expected = inputs + (inputs * hidden) + (hidden * outputs)
    XCTAssertTrue(flattenedWeights.count == expected,
                  "got: \(flattenedWeights.count) expected: \(expected)")
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
  
  func testFeedIsntSame() {
    var previous: [Float] = [Float](repeating: 0.0, count: self.inputs)
    
    for i in 0..<10 {
      var inputs: [Float] = []
      for _ in 0..<self.inputs {
        inputs.append(Float.random(in: 0...1))
      }
      
      let out = self.brain.feed(input: inputs)
      
      print("Feed \(i): \(out)")
      XCTAssertTrue(previous != out, "Result is the same check code...")
      previous = out
    }
    
  }
  
  func testTraining() {
    print("Training....")
    self.brain.train(data: self.trainingData, validation: self.validationData) { (complete) in
      let lastFive = self.brain.loss[self.brain.loss.count - 5..<self.brain.loss.count]
      var sum: Float = 0
      lastFive.forEach { (last) in
        sum += last
      }
      let average = sum / 5
      XCTAssertTrue(average <= self.testingLossThreshold, "Network did not learn, average loss was \(average)")
    }
    
    for i in 0..<ColorType.allCases.count {
      let color = ColorType.allCases[i]
      
      let out = self.brain.feed(input: color.color())
      print("Guess \(color.string): \(out)")
      
      XCTAssert(out.max() != nil, "No max value. Training failed")

      if let max = out.max(), let first = out.firstIndex(of: max) {
        XCTAssertTrue(first == i, "Color \(color.string) could not be identified")
      }
    }
  }
}

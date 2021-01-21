import XCTest
@testable import Neuron

final class NeuronPretrainedClassificationTests: XCTestCase, ModelBuilder {

  let inputs = 4 //UIColor values rgba
  let hidden = 5
  let outputs = ColorType.allCases.count
  let numOfHiddenLayers = 1
  
  let lossThreshold: Float = 0.001
  let testingLossThreshold: Float = 0.2 //if below 0.2 considered trained
  
  static var allTests = [
    ("testCNeuronConnectionObjects", testCNeuronConnectionObjects),
    ("testWeightNumbers", testBWeightNumbers),
    ("testANumberOfLobesMatches", testANumberOfLobesMatches),
    ("testFeedIsntSame", testDFeedIsntSame),
    ("testGuess", testEGuess),

  ]
  
  private var brain: Brain?
  
  public var trainingData: [TrainingData] = []
  public var validationData: [TrainingData] = []
  
  override func setUp() {
    super.setUp()
    
    self.importPretrainedModel()
        
    XCTAssertTrue(brain != nil, "Brain is empty")
    
    if let brain = brain {
      XCTAssertTrue(brain.compiled, "Brain is not compiled")
    }
  }
  
  func importPretrainedModel() {
    if let modelJson: [AnyHashable: Any]? = self.getJSON("sample_color_model", "smodel"),
       let model: ExportModel = self.build(modelJson) {
     
      let brain = Brain(model: model,
                        epochs: 100,
                        lossFunction: .crossEntropy,
                        lossThreshold: 0.001,
                        initializer: .xavierUniform)
      
      brain.add(modifier: .softmax)
      self.brain = brain
    }
  }
 
  func testANumberOfLobesMatches() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    guard let brain = self.brain else {
      return
    }
    
    let inputLayer = brain.lobes.filter({ $0.layer == .input })
    let hiddenLayers = brain.lobes.filter({ $0.layer == .hidden })
    let outputLayer = brain.lobes.filter({ $0.layer == .output })

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
  
  func testBWeightNumbers() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    guard let brain = self.brain else {
      return
    }
  
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
  func testCNeuronConnectionObjects() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    guard let brain = self.brain else {
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
  
  func testDFeedIsntSame() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    guard let brain = self.brain else {
      return
    }
    
    var previous: [Float] = [Float](repeating: 0.0, count: self.inputs)
    
    for i in 0..<10 {
      var inputs: [Float] = []
      for _ in 0..<self.inputs {
        inputs.append(Float.random(in: 0...1))
      }
      
      let out = brain.feed(input: inputs)
      
      print("Feed \(i): \(out)")
      XCTAssertTrue(previous != out, "Result is the same check code...")
      previous = out
    }
    
  }
  
  func testEGuess() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    guard let brain = self.brain else {
      return
    }
    
    for i in 0..<ColorType.allCases.count {
      let color = ColorType.allCases[i]
      
      let out = brain.feed(input: color.color())
      print("Guess \(color.string): \(out)")
      
      XCTAssert(out.max() != nil, "No max value. Training failed")

      if let max = out.max(), let first = out.firstIndex(of: max) {
        XCTAssertTrue(first == i, "Color \(color.string) could not be identified")
      }
    }
  }
  
}

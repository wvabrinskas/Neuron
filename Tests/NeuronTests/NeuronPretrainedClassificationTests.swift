import XCTest
@testable import Neuron

extension Array where Element: Comparable {
  func compare(_ compare: [Element]) -> Bool {
    guard self.count == compare.count else {
      return false
    }
    
    var returnValue = false
    
    for i in 0..<self.count {
      let selfVal = self[i]
      let compVal = compare[i]
      returnValue = selfVal == compVal
    }
    
    return returnValue
  }
}

final class NeuronPretrainedClassificationTests: XCTestCase, BaseTestConfig {

  static var allTests = [
    ("testWeightNumbers", testWeightNumbers),
    ("testANumberOfLobesMatches", testNumberOfLobesMatches),
    ("testGuess", testGuess),
  ]
  
  public var brain: Brain?
  public var model: ExportModel?
  
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
    do {
      let fileURL = try Resource(name: TestConstants.samplePretrainedModel, type: "smodel").url
      
      let model = PretrainedModel(url: fileURL)
      
      let brain = Brain(model: model,
                        epochs: 200,
                        lossFunction: .crossEntropy,
                        lossThreshold: TestConstants.lossThreshold,
                        initializer: .xavierNormal)
      
      brain?.add(modifier: .softmax)
      self.brain = brain
      self.model = brain?.model
      
      brain?.compile()
      
    } catch {
      print(error)
    }
  }
  
  func testWeightsInModel() {
    XCTAssertTrue(model != nil, "No model to test against")
    XCTAssertTrue(brain != nil, "Brain is empty")

    guard let model = model, let brain = brain else {
      return
    }
    
    let allWeights = model.layers.map({ $0.weights })
    
    for l in 0..<brain.lobes.count {
      let lobe = brain.lobes[l]
      let weightsAtLayer = lobe.neurons.map({ $0.inputs.map({ $0.weight} )})
      let weightsAtPretrained = allWeights[l]
      
      for i in 0..<weightsAtLayer.count {
        let weight = weightsAtLayer[i]
        let weightP = weightsAtPretrained[i]
        
        XCTAssertTrue(weight.compare(weightP), "Layer weights don't match")
      }
    }

  }
 
  func testNumberOfLobesMatches() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    guard let brain = self.brain else {
      return
    }
    
    let inputLayer = brain.lobes.filter({ $0.layer == .input })
    let hiddenLayers = brain.lobes.filter({ $0.layer == .hidden })
    let outputLayer = brain.lobes.filter({ $0.layer == .output })

    XCTAssertTrue(inputLayer.count == 1, "Should only have 1 input layer")

    if let first = inputLayer.first {
      XCTAssertTrue(first.neurons.count == TestConstants.inputs, "Input layer count does not match model")
    }
    
    XCTAssertTrue(hiddenLayers.count == TestConstants.numOfHiddenLayers, "Number of hidden layers does not match model")
    
    hiddenLayers.forEach { (layer) in
      XCTAssertTrue(layer.neurons.count == TestConstants.hidden, "Hidden layer count does not match model")
    }
    
    XCTAssertTrue(outputLayer.count == 1, "Should only have 1 output layer")

    if let first = outputLayer.first {
      XCTAssertTrue(first.neurons.count == TestConstants.outputs, "Output layer count does not match model")
    }
    
  }
  
  func testWeightNumbers() {
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
    
    let expected = TestConstants.inputs + (TestConstants.inputs * TestConstants.hidden) + (TestConstants.hidden * TestConstants.outputs)
    XCTAssertTrue(flattenedWeights.count == expected,
                  "got: \(flattenedWeights.count) expected: \(expected)")
  }
  
  func testGuess() {
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

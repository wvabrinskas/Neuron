import XCTest
@testable import Neuron


final class NeuronClassificationTests:  XCTestCase, BaseTestConfig, ModelBuilder {

  public lazy var brain: Brain? = {
    let bias: Float = 0//0.00001
    
    let brain = Brain(learningRate: 0.001,
                      epochs: 1000,
                      lossFunction: .crossEntropy,
                      lossThreshold: TestConstants.lossThreshold,
                      initializer: .xavierNormal,
                      descent: .mbgd(size: 32))
    
    brain.addInputs(TestConstants.inputs)
    
    for _ in 0..<TestConstants.numOfHiddenLayers {
//      let normalLobe = NormalizedLobeModel(nodes: TestConstants.hidden,
//                                           activation: .leakyRelu,
//                                           bias: bias,
//                                           momentum: 0.99,
//                                           normalizerLearningRate: 0.1)
//      brain.add(normalLobe)
      brain.add(LobeModel(nodes: TestConstants.hidden, activation: .leakyRelu, bias: bias)) //hidden layer
    }
    
    brain.add(LobeModel(nodes: TestConstants.outputs, activation: .none, bias: bias)) //output layer
    
    brain.add(modifier: .softmax) //when using softmax activation the output node should use a reLu or leakyRelu activation
    
   // brain.add(optimizer: .adam())
    brain.logLevel = .low
    
    return brain
  }()
  
  public var trainingData: [TrainingData] = []
  public var validationData: [TrainingData] = []
  
  override func setUp() {
    super.setUp()
    
    XCTAssertTrue(brain != nil, "Brain is empty")
    
    guard let brain = brain else {
      return
    }

    if !brain.compiled {
      print("setting up")
      brain.compile()
      XCTAssertTrue(brain.compiled, "Brain not initialized")
      
      self.buildTrainingData()
    }
  }
  
  func buildTrainingData() {
    let num = 600

    for _ in 0..<num {
      trainingData.append(TrainingData(data: ColorType.red.color(), correct: ColorType.red.correctValues()))
      validationData.append(TrainingData(data: ColorType.red.color(), correct: ColorType.red.correctValues()))
      trainingData.append(TrainingData(data: ColorType.green.color(), correct: ColorType.green.correctValues()))
      validationData.append(TrainingData(data: ColorType.green.color(), correct: ColorType.green.correctValues()))
      trainingData.append(TrainingData(data: ColorType.blue.color(), correct: ColorType.blue.correctValues()))
      validationData.append(TrainingData(data: ColorType.blue.color(), correct: ColorType.blue.correctValues()))
    }
    
  }
  
  //MARK: I really dont think we need to test training this is more of a test when building new architecture into the framework
  /// Uncomment out if you want to run a test training with out integrating into an app
  func testTraining() {
    XCTAssertTrue(brain != nil, "Brain is empty")

    guard let brain = brain else {
      return
    }

    print("Training....")
    let expectation = XCTestExpectation()

    brain.train(data: self.trainingData.randomize(),
                validation: self.validationData,
                complete:  { (complete) in
      expectation.fulfill()
    })

    wait(for: [expectation], timeout: 40)

    for i in 0..<ColorType.allCases.count {
      let color = ColorType.allCases[i]

      let out = brain.feed(input: color.color())
      print("Guess \(color.string): \(out)")

      XCTAssert(out.max() != nil, "No max value. Training failed")

      let max = out.max
      if let first = out.firstIndex(of: max) {
        XCTAssert(max.isNaN == false, "Result was NaN")
        XCTAssertTrue(first == i, "Color \(color.string) could not be identified")
      } else {
        XCTFail("No color to be found...")
      }
    }
  }
//
  //executes in alphabetical order
  func testXport() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    
    guard let brain = brain else {
      return
    }
      
    let url = brain.exportModelURL()
    print("📄 model: \(String(describing: url))")
    XCTAssertTrue(url != nil, "Could not build exported model")
  }
}

import XCTest
@testable import Neuron


final class NeuronClassificationTests:  XCTestCase, BaseTestConfig, ModelBuilder {
    
  static var allTests = [
    ("testTraining", testTraining),
    ("testXport", testXport)
  ]
  
  public lazy var brain: Brain? = {
    let bias: Float = 0.001
    
    let brain = Brain(learningRate: 0.01,
                      epochs: 200,
                      lossFunction: .crossEntropy,
                      lossThreshold: TestConstants.lossThreshold,
                      initializer: .xavierNormal,
                      descent: .sgd)
    
    brain.add(.init(nodes: TestConstants.inputs, bias: bias, normalize: false)) //input layer
    
    for _ in 0..<TestConstants.numOfHiddenLayers {
      brain.add(.init(nodes: TestConstants.hidden, activation: .reLu, bias: bias, normalize: false)) //hidden layer
    }
    
    brain.add(.init(nodes: TestConstants.outputs, bias: bias, normalize: false)) //output layer
    
    brain.add(modifier: .softmax)
    
    brain.add(optimizer: .adam())
    brain.logLevel = .none
    
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
    let num = 200

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

 
  func testTraining() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    
    guard let brain = brain else {
      return
    }
    
    print("Training....")
    
    brain.train(data: self.trainingData, validation: self.validationData) { (complete) in
      let lastFive = brain.loss[brain.loss.count - 5..<brain.loss.count]
      var sum: Float = 0
      lastFive.forEach { (last) in
        sum += last
      }
      let average = sum / 5
      XCTAssertTrue(average <= TestConstants.testingLossThreshold, "Network did not learn, average loss was \(average)")
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
  
//  func testXportLoss() {
//    XCTAssertTrue(brain != nil, "Brain is empty")
//    
//    guard let brain = brain else {
//      return
//    }
//    
//    print("Training for loss export....")
//    
//    brain.train(data: self.trainingData, validation: self.validationData) { (complete) in
//      let lastFive = brain.loss[brain.loss.count - 5..<brain.loss.count]
//      var sum: Float = 0
//      lastFive.forEach { (last) in
//        sum += last
//      }
//      let average = sum / 5
//      XCTAssertTrue(average <= TestConstants.testingLossThreshold, "Network did not learn, average loss was \(average)")
//    }
//    
//    for i in 0..<ColorType.allCases.count {
//      let color = ColorType.allCases[i]
//      
//      let out = brain.feed(input: color.color())
//      print("Guess \(color.string): \(out)")
//      
//      XCTAssert(out.max() != nil, "No max value. Training failed")
//
//      if let max = out.max(), let first = out.firstIndex(of: max) {
//        XCTAssertTrue(first == i, "Color \(color.string) could not be identified")
//      }
//    }
//    
//    let url = brain.exportLoss()
//    print("📉 loss: \(url)")
//    XCTAssertTrue(url != nil, "Could not build exported model")
//  }
  
  //executes in alphabetical order
  func testXport() {
    XCTAssertTrue(brain != nil, "Brain is empty")
    
    guard let brain = brain else {
      return
    }

    print("Training for export....")
    
    brain.train(data: self.trainingData, validation: self.validationData) { (complete) in
      let lastFive = brain.loss[brain.loss.count - 5..<brain.loss.count]
      var sum: Float = 0
      lastFive.forEach { (last) in
        sum += last
      }
      let average = sum / 5
      XCTAssertTrue(average <= TestConstants.testingLossThreshold, "Network did not learn, average loss was \(average)")
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
    
    let url = brain.exportModelURL()
    print("📄 model: \(url)")
    XCTAssertTrue(url != nil, "Could not build exported model")
  }
}

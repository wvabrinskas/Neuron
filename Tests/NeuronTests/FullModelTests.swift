//
//  File.swift
//
//
//  Created by William Vabrinskas on 5/23/22.
//

import Foundation
import XCTest
import NumSwift
@testable import Neuron

class MockRNNDataset: VectorizableDataset<String> {
  private let inputStrings: [String]
  private let labelStrings: [String]
  private let maxLength: Int
  private let trainingCount: Int
  private let validationCount: Int
  
  init(inputStrings: [String], 
       labelStrings: [String]? = nil,
       trainingCount: Int = 256,
       validationCount: Int = 10) {
    self.inputStrings = inputStrings
    // If labels not provided, derive by removing first character
    self.labelStrings = labelStrings ?? inputStrings.map { 
      $0.count > 1 ? String($0.dropFirst()) : $0 
    }
    self.maxLength = inputStrings.max(by: { $0.count < $1.count })?.count ?? 0
    self.trainingCount = trainingCount
    self.validationCount = validationCount
    
    super.init()
  }
  
  required init(vectorizer: Vectorizer<String> = .init()) {
    fatalError("init(vectorizer:) has not been implemented")
  }
  
  override func build() async -> Neuron.VectorizingDatasetData {
    // First pass: vectorize all inputs to build vocabulary
    for inputString in inputStrings {
      vectorizer.vectorize(inputString.fill(with: ".",
                                            max: maxLength).characters)
    }
    
    // Process each input-label pair
    var inputTensors: [Tensor] = []
    var labelTensors: [Tensor] = []
    
    for (inputString, labelString) in zip(inputStrings, labelStrings) {
      let oneHot = vectorizer.oneHot(labelString.fill(with: ".",
                                                      max: maxLength).characters)
      let input = vectorize(inputString.fill(with: ".",
                                             max: maxLength).characters)
      
      inputTensors.append(input)
      labelTensors.append(oneHot)
    }
    
    self.vocabSize = vectorizer.inverseVector.count
    // Build training and validation datasets
    var training: [DatasetModel] = []
    var val: [DatasetModel] = []
    
    for (inputTensor, labelTensor) in zip(inputTensors, labelTensors) {
      training.append(contentsOf: [DatasetModel](repeating: DatasetModel(data: inputTensor,
                                                                        label: labelTensor), 
                                                count: trainingCount))
      val.append(contentsOf: [DatasetModel](repeating: DatasetModel(data: inputTensor,
                                                                    label: labelTensor), 
                                           count: validationCount))
    }
    
    return (training, val)
  }
}

final class FullModelTests: XCTestCase {
  public var trainingData: [DatasetModel] = []
  public var validationData: [DatasetModel] = []
  
  override func setUp() {
    super.setUp()
    buildTrainingData()
  }
  
  func buildTrainingData() {
    let num = 6000
    for _ in 0..<num {
      let num = Int.random(in: 0...2)
      
      if num == 0 {
        trainingData.append(DatasetModel(data: Tensor(ColorType.red.color()),
                                         label: Tensor(ColorType.red.correctValues())))
        validationData.append(DatasetModel(data: Tensor(ColorType.red.color()),
                                           label: Tensor(ColorType.red.correctValues())))
      } else if num == 1 {
        trainingData.append(DatasetModel(data: Tensor(ColorType.green.color()),
                                         label: Tensor(ColorType.green.correctValues())))
        validationData.append(DatasetModel(data: Tensor(ColorType.green.color()),
                                           label: Tensor(ColorType.green.correctValues())))
      } else if num == 2 {
        trainingData.append(DatasetModel(data: Tensor(ColorType.blue.color()),
                                         label: Tensor(ColorType.blue.correctValues())))
        validationData.append(DatasetModel(data: Tensor(ColorType.blue.color()),
                                           label: Tensor(ColorType.blue.correctValues())))
      }
    }
    
  }
  
  func test_resNet_in_Network() {
    
    let inputSize: TensorSize = .init(rows: 28, columns: 28, depth: 1)
    let classes = 10
    
    let network = Sequential {
      [
        Conv2d(filterCount: 32, inputSize: inputSize, strides: (1,1), padding: .same, filterSize: (3,3)),
        BatchNormalize(),
        ReLu(),
        ResNet(filterCount: 32, stride: 1),
        ResNet(filterCount: 64, stride: 2),
        GlobalAvgPool(),
        Dense(classes, biasEnabled: true),
        Softmax()
      ]
    }
    
    network.compile()
    
    network.isTraining = true
    
    let input = Tensor.fillRandom(size: inputSize)
    
    let out = network(input, context: .init())
    
    XCTAssertEqual(TensorSize(array: out.shape), .init(array: [classes,1,1]))
    
    let error = Tensor.fillRandom(size: .init(array: [classes,1,1]))
    
    let gradient = out.gradients(delta: error, wrt: input)
    
    XCTAssertEqual(gradient.input.count, network.layers.count)
    
    network.apply(gradients: gradient, learningRate: 0.001)
  }
  
  func test_setttingPropertyOnTrainable_Doesnt_Reset() {
    let network = Sequential {
      [
        Dense(1, inputs: 2)
      ]
    }
    
    let optim = Adam(network, learningRate: 0.01, batchSize: 1)
    
    network.isCompiled = false
    
    // testing that when setting this it doesn't attempt to recompile the network
    optim.isTraining = false
    
    XCTAssertEqual(network.isCompiled, false)
  }
  
  func testBasicClassification() {
    let batchSize = 32
    
    let network = Sequential {
      [
        Dense(12, inputs: 4,
              initializer: .xavierUniform,
              biasEnabled: true),
        ReLu(),
        BatchNormalize(),
        Dense(3, initializer: .xavierUniform),
        Softmax()
      ]
    }
    
    let optim = Adam(network, learningRate: 0.01, batchSize: batchSize)
    
    let reporter = MetricsReporter(metricsToGather: [.loss,
                                                     .accuracy,
                                                     .valAccuracy,
                                                     .valLoss])
    
    optim.metricsReporter = reporter
    
    let classifier = Classifier(optimizer: optim,
                                batchSize: batchSize, // 64 or higher causes issuees with BatchNorm for some reason.
                                accuracyThreshold: .init(value: 0.9, averageCount: 3))
    
    classifier.onAccuracyReached = {
      let red = ColorType.red.color()
      let green = ColorType.green.color()
      let blue = ColorType.blue.color()
      
      let colors = [Tensor(red), Tensor(green), Tensor(blue)]
      let out = classifier.feed(colors)
      
      for i in 0..<out.count {
        let o = out[i]
        XCTAssert(o.storage.indexOfMax.0 == i)
      }
    }
    
    classifier.fit(trainingData, validationData)
  }
  
  func testImportPretrainedClassifier() {
    let batchSize = 64
    
    do {
      let fileURL = try Resource(name: "pretrained-classifier-color", type: ExportHelper.FileExtensions.smodel.rawValue).url
      
      let n = Sequential.import(fileURL)
      let optim = Adam(n, learningRate: 0.0001, batchSize: batchSize)
      
      let classifier = Classifier(optimizer: optim,
                                  batchSize: batchSize,
                                  accuracyThreshold: .init(value: 0.9, averageCount: 5))
      
      let red = ColorType.red.color()
      let green = ColorType.green.color()
      let blue = ColorType.blue.color()
      
      let colors = [Tensor(red), Tensor(green), Tensor(blue)]
      let out = classifier.feed(colors)
      
      for i in 0..<out.count {
        let o = out[i]
        XCTAssert(o.storage.indexOfMax.0 == i)
      }
      
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  /// LSTM test example
  func test_LSTM_Forward_Example() async {
    guard isGithubCI == false else {
      XCTAssertTrue(true)
      return
    }
    
    let inputUnits = 64
    let hiddenUnits = 256
    
    let reporter = MetricsReporter(frequency: 1,
                                   metricsToGather: [.loss,
                                                     .accuracy,
                                                     .valAccuracy,
                                                     .valLoss])


    let dataset = MockRNNDataset(inputStrings: ["hammley",
                                                "spammley",
                                                "Dugley",
                                                "Absoluteley"])
    
    let rnn = RNN(returnSequence: true,
                  dataset: dataset,
                  classifierParameters: RNN.ClassifierParameters(batchSize: 256,
                                                                 epochs: 60,
                                                                 accuracyThreshold: .init(value: 0.95, averageCount: 5),
                                                                 killOnAccuracy: true),
                  optimizerParameters: RNN.OptimizerParameters(learningRate: 0.0006,
                                                               weightDecay: .decay(0.0001),
                                                               metricsReporter: reporter),
                  lstmParameters: RNN.RNNLSTMParameters(hiddenUnits: hiddenUnits,
                                                        inputUnits: inputUnits,
                                                        embeddingInitializer: .orthogonal(gain: 1.0),
                                                        lstmInitializer: .orthogonal(gain: 1.0)))
    
    
    reporter.receive = { _ in }
    
    rnn.onEpochCompleted = {
      let r = rnn.predict(count: 10, maxWordLength: 20, randomizeSelection: false)
      print(r)
      
      let exports = rnn.exportWithVectors(overrite: true, compress: true)
      
      print(exports)
      
      // let s = rnn.predict(count: 10, maxWordLength: 20, randomizeSelection: false)
      // print(s)
    }
    
    await rnn.train()
  }
}

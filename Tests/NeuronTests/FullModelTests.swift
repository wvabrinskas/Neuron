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


class MockRNNDataset: RNNSupportedDataset {
  var vocabSize: Int = 0
  
  func oneHot(_ items: [String]) -> Neuron.Tensor {
    vectorizer.oneHot(items)
  }
  
  let vectorizer = Vectorizer<String>()
  
  func getWord(for data: Neuron.Tensor) -> [String] {
    return vectorizer.unvectorizeOneHot(data)
  }
  
  func build() async -> Neuron.RNNSupportedDatasetData {
    let max = 9
    vectorizer.vectorize("hammley".fill(with: ".",
                                     max: max).characters)
    
    vectorizer.vectorize("spammley".fill(with: ".",
                                         max: max).characters)
    
    let oneHot = vectorizer.oneHot("ammley".fill(with: ".",
                                                  max: max).characters)
    let oneHot2 = vectorizer.oneHot("pammley".fill(with: ".",
                                                  max: max).characters)
    
    let input = vectorizer.oneHot("hammley".fill(with: ".",
                                                  max: max).characters)
    let input2 = vectorizer.oneHot("spammley".fill(with: ".",
                                                  max: max).characters)
    
    vocabSize = vectorizer.inverseVector.count
    
    let labelTensor = oneHot
    let inputTensor = input
    
    let labelTensor2 = oneHot2
    let inputTensor2 = input2
    
    var training = [DatasetModel](repeating: DatasetModel(data: inputTensor,
                                                          label: labelTensor), count: 1000)
    var val = [DatasetModel](repeating: DatasetModel(data: inputTensor,
                                                      label: labelTensor), count: 10)
    
    training.append(contentsOf: [DatasetModel](repeating: DatasetModel(data: inputTensor2,
                                                                       label: labelTensor2), count: 1000))
    
    val.append(contentsOf: [DatasetModel](repeating: DatasetModel(data: inputTensor2,
                                                                  label: labelTensor2), count: 10))
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
        XCTAssert(o.value[safe: 0, [[0]]][safe: 0, [0]].indexOfMax.0 == i)
      }
    }
    
    classifier.fit(trainingData, validationData)
  }
  
  func testImportPretrainedClassifier() {
    let batchSize = 64
    
    do {
      let fileURL = try Resource(name: "pretrained-classifier-color", type: "smodel").url
      
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
        XCTAssert(o.value[safe: 0, [[0]]][safe: 0, [0]].indexOfMax.0 == i)
      }
      
    } catch {
      XCTFail(error.localizedDescription)
    }
  }
  
  /// LSTM test example
  func test_LSTM_Forward_Example() async {
    // guard isGithubCI == false else {
    //   XCTAssertTrue(true)
    //   return
    // }

    let inputUnits = 50
    let hiddenUnits = 100
    
    let reporter = MetricsReporter(frequency: 1,
                                   metricsToGather: [.loss,
                                                     .accuracy,
                                                     .valAccuracy,
                                                     .valLoss])

    
    let rnn = RNN(returnSequence: true,
                  dataset: MockRNNDataset(),
                  classifierParameters: RNN.ClassifierParameters(batchSize: 16,
                                                                 epochs: 1000,
                                                                 accuracyThreshold: .init(value: 0.8, averageCount: 5),
                                                                 killOnAccuracy: false),
                  optimizerParameters: RNN.OptimizerParameters(learningRate: 0.002,
                                                               gradientClipping: 5.0,
                                                               metricsReporter: reporter),
                  lstmParameters: RNN.RNNLSTMParameters(hiddenUnits: hiddenUnits,
                                                        inputUnits: inputUnits))
    
    
    reporter.receive = { _ in }
        
    rnn.onEpochCompleted = {
      let r = rnn.predict(starting: "h", maxWordLength: 20, randomizeSelection: false)
      print(r)
      let s = rnn.predict(starting: "s", maxWordLength: 20, randomizeSelection: false)
      print(s)
    }
    
    await rnn.train()
  }
}

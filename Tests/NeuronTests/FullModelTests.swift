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
  func oneHot(_ items: [String]) -> Neuron.Tensor {
    vectorizer.oneHot(items)
  }
  
  let vectorizer = Vectorizer<String>()
  
  func getWord(for data: Neuron.Tensor) -> [String] {
    return vectorizer.unvectorizeOneHot(data)
  }
  
  func build() async -> Neuron.RNNSupportedDatasetData {
    vectorizer.vectorize("xavier".fill(with: ".",
                                     max: 10).characters)
    
    let oneHot = vectorizer.oneHot("xavier".fill(with: ".",
                                               max: 7).characters)
    
    var labels: [[[Float]]] = Array(oneHot.value.dropFirst())
    labels.append([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    
    let labelTensor = Tensor(labels)
    let inputTensor = oneHot
    
    return ([DatasetModel](repeating: DatasetModel(data: inputTensor,
                                                  label: labelTensor), count: 900),
           [DatasetModel](repeating: DatasetModel(data: inputTensor,
                                                  label: labelTensor), count: 5))
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
  
  func testBasicClassification() {
    let network = Sequential {
      [
        Dense(6, inputs: 4, initializer: .xavierNormal),
        LeakyReLu(limit: 0.2),
        //BatchNormalize(), //-> removed since sometimes during tests it can crash
        Dense(3, initializer: .xavierNormal),
        Softmax()
      ]
    }
    
    let optim = Adam(network, learningRate: 0.01)
    
    let reporter = MetricsReporter(metricsToGather: [.loss,
                                                     .accuracy,
                                                     .valAccuracy,
                                                     .valLoss])
    
    optim.metricsReporter = reporter
    
    let classifier = Classifier(optimizer: optim,
                                batchSize: 64,
                                accuracyThreshold: 0.9)

    optim.metricsReporter?.receive = { metrics in
      //let accuracy = metrics[.accuracy] ?? 0
      //let loss = metrics[.loss] ?? 0
      //let valLoss = metrics[.valLoss] ?? 0
      
      //print("training -> ", "loss: ", loss, "val_loss: ", valLoss, "accuracy: ", accuracy)
    }
    
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
    do {
      let fileURL = try Resource(name: "pretrained-classifier-color", type: "smodel").url
      
      let n = Sequential.import(fileURL)
      let optim = Adam(n, learningRate: 0.0001)
      
      let classifier = Classifier(optimizer: optim,
                                  batchSize: 64,
                                  accuracyThreshold: 0.9)
      
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
    guard isGithubCI == false else {
      XCTAssertTrue(true)
      return
    }

    let inputUnits = 25
    let hiddenUnits = 100
    
    let reporter = MetricsReporter(frequency: 1,
                                   metricsToGather: [.loss,
                                                     .accuracy,
                                                     .valAccuracy,
                                                     .valLoss])

    
    let rnn = RNN(returnSequence: true,
                  dataset: MockRNNDataset(),
                  classifierParameters: RNN.ClassifierParameters(batchSize: 16,
                                                                 epochs: 40,
                                                                 accuracyThreshold: 0.8,
                                                                 killOnAccuracy: false,
                                                                 threadWorkers: 8),
                  optimizerParameters: RNN.OptimizerParameters(learningRate: 0.005,
                                                               metricsReporter: reporter),
                  lstmParameters: RNN.RNNLSTMParameters(hiddenUnits: hiddenUnits,
                                                       inputUnits: inputUnits)) 
    
    
    reporter.receive = { metrics in
      let accuracy = metrics[.accuracy] ?? 0
      let loss = metrics[.loss] ?? 0
      //print("training -> ", "loss: ", loss, "accuracy: ", accuracy)
    }
        
    rnn.onEpochCompleted = {
      let r = rnn.predict()
      print(r)
    }
    
    await rnn.train()

//
//    rnn.onAccuracyReached = {
//
//    }
//      
//    await rnn.readyUp()
//
//    let r = rnn.predict()
//    print(r)
  }
}

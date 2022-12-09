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
  
//  func testMNISTClassifier() async {
//    guard isGithubCI == false else {
//      XCTAssert(true)
//      return
//    }
//
//    let initializer: InitializerType = .heNormal
//    
//    let flatten = Flatten()
//    flatten.inputSize = TensorSize(array: [28, 28, 1])
//    
//    let network = Sequential {
//      [
//        Conv2d(filterCount: 16,
//               inputSize: TensorSize(array: [28,28,1]),
//               padding: .same,
//               initializer: initializer),
//        BatchNormalize(),
//        LeakyReLu(limit: 0.2),
//        MaxPool(),
//        Conv2d(filterCount: 32,
//               padding: .same,
//               initializer: initializer),
//        BatchNormalize(),
//        LeakyReLu(limit: 0.2),
//        Dropout(0.5),
//        MaxPool(),
//        Flatten(),
//        Dense(64, initializer: initializer),
//        LeakyReLu(limit: 0.2),
//        Dense(10, initializer: initializer),
//        Softmax()
//      ]
//    }
//    
//    let optim = Adam(network, learningRate: 0.0001, l2Normalize: false)
//    
//    let reporter = MetricsReporter(frequency: 1,
//                                   metricsToGather: [.loss,
//                                                     .accuracy,
//                                                     .valAccuracy,
//                                                     .valLoss])
//    
//    optim.metricsReporter = reporter
//    
//    optim.metricsReporter?.receive = { metrics in
//      let accuracy = metrics[.accuracy] ?? 0
//      let loss = metrics[.loss] ?? 0
//      //let valLoss = metrics[.valLoss] ?? 0
//      
//      print("training -> ", "loss: ", loss, "accuracy: ", accuracy)
//    }
//    
//    let classifier = Classifier(optimizer: optim,
//                                epochs: 10,
//                                batchSize: 32,
//                                threadWorkers: 8,
//                                log: false)
//    
//    let data = await MNIST().build()
//    
//    /*
//     for _ in 0..<3 {
//       if let random = data.val.randomElement() {
//         let label = random.label
//         let data = random.data
//         
//         let labelArray: [Float] = label.value.flatten()
//         let labelMax = labelArray.indexOfMax.0
//         
//         let out = classifier.feed([data])
//         if let first = out.first {
//           let firstFlat: [Float] = first.value.flatten()
//           let firstMax = firstFlat.indexOfMax.0
//           
//           print("out: ", firstMax, "- \(firstFlat.indexOfMax.1 * 100)%", "---> expected: ", labelMax, "- \(labelArray.indexOfMax.1 * 100)%")
//           XCTAssert(firstMax == labelMax)
//         }
//       }
//     }
//     */
//    classifier.fit(data.training, data.val)
//  }
}

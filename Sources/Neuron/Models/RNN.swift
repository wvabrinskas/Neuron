//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/28/23.
//

import Foundation
import NumSwift

public typealias RNNSupportedDatasetData = (training: [DatasetModel], val: [DatasetModel])
public protocol RNNSupportedDataset {
  func getWord<T: RandomAccessCollection>(for data: T) -> [String]
  func build() async -> RNNSupportedDatasetData
}

public class RNN: Classifier {
  public struct RNNLSTMParameters {
    let hiddenUnits: Int
    let vocabSize: Int
    let wordLength: Int
    let inputUnits: Int
    
    public init(hiddenUnits: Int,
                vocabSize: Int,
                wordLength: Int,
                inputUnits: Int) {
      self.hiddenUnits = hiddenUnits
      self.vocabSize = vocabSize
      self.wordLength = wordLength
      self.inputUnits = inputUnits
    }
  }
  
  public struct OptimizerParameters {
    let learningRate: Float
    let b1: Float
    let b2: Float
    let eps: Float
    let metricsReporter: MetricsReporter?
    
    public init(learningRate: Float,
                b1: Float = 0.9,
                b2: Float = 0.999,
                eps: Float = 1e-8,
                metricsReporter: MetricsReporter? = nil) {
      self.learningRate = learningRate
      self.b1 = b1
      self.b2 = b2
      self.eps = eps
      self.metricsReporter = metricsReporter
    }
  }
  
  public struct ClassifierParameters {
    let batchSize: Int
    let epochs: Int
    let accuracyThreshold: Float
    let killOnAccuracy: Bool
    let threadWorkers: Int
    
    public init(batchSize: Int,
                epochs: Int,
                accuracyThreshold: Float = 0.9,
                killOnAccuracy: Bool = true,
                threadWorkers: Int = 8) {
      self.batchSize = batchSize
      self.epochs = epochs
      self.accuracyThreshold = accuracyThreshold
      self.killOnAccuracy = killOnAccuracy
      self.threadWorkers = threadWorkers
    }
  }
  
  private let dataset: RNNSupportedDataset
  private let lstm: LSTM
  
  private let classifierParameters: ClassifierParameters
  private let optimizerParameters: OptimizerParameters
  private let lstmParameters: RNNLSTMParameters
  
  public init(device: Device = CPU(),
              dataset: RNNSupportedDataset,
              classifierParameters: ClassifierParameters,
              optimizerParameters: OptimizerParameters,
              lstmParameters: RNNLSTMParameters,
              extraLayers: () -> [Layer] = { [] }) {
    
    self.classifierParameters = classifierParameters
    self.optimizerParameters = optimizerParameters
    self.lstmParameters = lstmParameters
    
    self.dataset = dataset
    
    self.lstm = LSTM(inputSize: TensorSize(rows: lstmParameters.wordLength,
                                           columns: lstmParameters.inputUnits,
                                           depth: 1),
                     initializer: .heNormal,
                     hiddenUnits: lstmParameters.hiddenUnits,
                     vocabSize: lstmParameters.vocabSize)
    
    var layers: [Layer] = [lstm]
    layers.append(contentsOf: extraLayers())
    
    let network = Sequential { layers }
    
    let optimizer = Adam(network,
                         device: device,
                         learningRate: optimizerParameters.learningRate,
                         b1: optimizerParameters.b1,
                         b2: optimizerParameters.b2,
                         eps: optimizerParameters.eps,
                         l2Normalize: false)
    
    optimizer.metricsReporter = optimizerParameters.metricsReporter
    
    super.init(optimizer: optimizer,
               epochs: classifierParameters.epochs,
               batchSize: classifierParameters.batchSize,
               accuracyThreshold: classifierParameters.accuracyThreshold,
               killOnAccuracy: classifierParameters.killOnAccuracy,
               threadWorkers: classifierParameters.threadWorkers,
               log: false)
  }
  
  public func train() async {
    let datasetOut = await dataset.build()
    fit(datasetOut.training, datasetOut.val)
  }
  
  public func predict(_ tensor: Tensor) -> String {
    var name: String = ""
    var runningChar: String = ""
    
    let vocabSize = lstmParameters.vocabSize
    
    var batch = [Float](repeating: 0, count: vocabSize)
    let index = Int.random(in: 0..<vocabSize)
    
    batch[index] = 1.0
    
    while runningChar != ".", name.count < 50 {
      // use LSTM layer since that's really all we need
      let out = lstm.forward(tensor: Tensor(batch))
      let outRounded = out.value.flatten()
      let maxIndex = Int(outRounded.indexOfMax.0)
      
      var outHot = [Float](repeating: 0, count: vocabSize)
      outHot[maxIndex] = 1.0
      
      let unvec = dataset.getWord(for: [outHot]).joined()
      
      name += unvec
      runningChar = unvec
      
      batch = outHot
    }
    
    return name
  }
}


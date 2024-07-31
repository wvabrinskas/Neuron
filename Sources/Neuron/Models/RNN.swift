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
  func oneHot(_ items: [String]) -> Tensor
  func getWord(for data: Tensor) -> [String]
  func build() async -> RNNSupportedDatasetData
}

public class RNN<N: TensorNumeric>: Classifier<N> {
  public struct RNNLSTMParameters {
    let hiddenUnits: Int
    let inputUnits: Int
    let embeddingInitializer: InitializerType
    let lstmInitializer: InitializerType

    public init(hiddenUnits: Int,
                inputUnits: Int,
                embeddingInitializer: InitializerType = .xavierNormal,
                lstmInitializer: InitializerType = .xavierNormal) {
      self.hiddenUnits = hiddenUnits
      self.inputUnits = inputUnits
      self.embeddingInitializer = embeddingInitializer
      self.lstmInitializer = lstmInitializer
    }
  }
  
  public struct OptimizerParameters {
    let learningRate: Tensor.Scalar
    let b1: Tensor.Scalar
    let b2: Tensor.Scalar
    let eps: Tensor.Scalar
    let weightDecay: Adam<N>.WeightDecay
    let metricsReporter: MetricsReporter?
    
    public init(learningRate: Tensor.Scalar,
                b1: Tensor.Scalar = 0.9,
                b2: Tensor.Scalar = 0.999,
                eps: Tensor.Scalar = .stabilityFactor,
                weightDecay: Adam<N>.WeightDecay = .none,
                metricsReporter: MetricsReporter? = nil) {
      self.learningRate = learningRate
      self.b1 = b1
      self.b2 = b2
      self.eps = eps
      self.metricsReporter = metricsReporter
      self.weightDecay = weightDecay
    }
  }
  
  public struct ClassifierParameters {
    let batchSize: Int
    let epochs: Int
    let accuracyThreshold: Tensor.Scalar
    let killOnAccuracy: Bool
    let threadWorkers: Int
    let lossFunction: LossFunction
    
    public init(batchSize: Int,
                epochs: Int,
                accuracyThreshold: Tensor.Scalar = 0.9,
                killOnAccuracy: Bool = true,
                threadWorkers: Int = 8,
                lossFunction: LossFunction = .binaryCrossEntropySoftmax) {
      self.batchSize = batchSize
      self.epochs = epochs
      self.accuracyThreshold = accuracyThreshold
      self.killOnAccuracy = killOnAccuracy
      self.threadWorkers = threadWorkers
      self.lossFunction = lossFunction
    }
  }
  
  private let dataset: RNNSupportedDataset
  private var lstm: LSTM<N>?
  private var embedding: Embedding<N>?
  private var vocabSize: Int = 0
  private var wordLength: Int = 0
  private var extraLayers: [BaseLayer<N>]
  private var ready: Bool = false
  private var datasetData: RNNSupportedDatasetData?
  private let returnSequence: Bool
  
  private let classifierParameters: ClassifierParameters
  private let optimizerParameters: OptimizerParameters
  private let lstmParameters: RNNLSTMParameters
  
  public init(device: Device = CPU(),
              returnSequence: Bool = true,
              dataset: RNNSupportedDataset,
              classifierParameters: ClassifierParameters,
              optimizerParameters: OptimizerParameters,
              lstmParameters: RNNLSTMParameters,
              extraLayers: () -> [BaseLayer<N>] = { [] }) {
    
    self.classifierParameters = classifierParameters
    self.optimizerParameters = optimizerParameters
    self.lstmParameters = lstmParameters
    
    self.returnSequence = returnSequence
    self.dataset = dataset
    self.extraLayers = extraLayers()
    
    let network = Sequential<N> { [] }
    
    let optimizer = Adam(network,
                         device: device,
                         learningRate: optimizerParameters.learningRate,
                         b1: optimizerParameters.b1,
                         b2: optimizerParameters.b2,
                         eps: optimizerParameters.eps,
                         weightDecay: optimizerParameters.weightDecay)
    
    optimizer.metricsReporter = optimizerParameters.metricsReporter
    
    super.init(optimizer: optimizer,
               epochs: classifierParameters.epochs,
               batchSize: classifierParameters.batchSize,
               accuracyThreshold: classifierParameters.accuracyThreshold,
               killOnAccuracy: classifierParameters.killOnAccuracy,
               threadWorkers: classifierParameters.threadWorkers,
               log: false,
               lossFunction: classifierParameters.lossFunction)
  }
  
  public func importFrom(url: URL?) async {
    guard let url else { return }
    
    await readyUp()
    
    let n = Sequential<N>.import(url)
    optimizer.trainable = n
  }
  
  public func train() async {
    optimizer.isTraining = true
    
    await readyUp()
    
    if let datasetData {
      fit(datasetData.training, datasetData.val)
    }
  }
  
  public func predict(starting with: String? = nil,
                      count: Int = 1,
                      maxWordLength: Int = 20,
                      randomizeSelection: Bool = false,
                      endingMark: String = ".") -> [String] {
    optimizer.isTraining = false
    
    var names: [String] = []
    
    for _ in 0..<count {
      
      var name: String = ""
      var runningChar: String = ""
          
      var batch: [[[Tensor.Scalar]]]
      
      if let with {
        let oneHotWith = dataset.oneHot([with])
        batch = oneHotWith.value
        name += with

      } else {
        var localWord = [Tensor.Scalar](repeating: 0, count: vocabSize)
        let index = Int.random(in: 0..<vocabSize)
        
        localWord[index] = 1.0
        
        batch = [[localWord]]
        
        // append random letter
        let unvec = dataset.getWord(for: Tensor(batch)).joined()
        name += unvec
      }

      while runningChar != endingMark && name.count < maxWordLength {
        
        let out = optimizer.predict([Tensor(batch)])
        
        // output: (col: vocabSize, rows: 1, depth: batchLength)
        guard let flat = out[safe: 0]?.value[safe: batch.count - 1]?.first else {
          break
        }
      
        var v: [Tensor.Scalar] = [Tensor.Scalar](repeating: 0, count: flat.count)
        
        let indexToChoose: Int
        if randomizeSelection {
          indexToChoose = NumSwift.randomChoice(in: Array(0..<vocabSize), p: flat).1
        } else {
          indexToChoose = Int(flat.indexOfMax.0)
        }
        
        v[indexToChoose] = 1
        
        let unvec = dataset.getWord(for: Tensor(v)).joined()
        
        runningChar = unvec
        name += unvec
        
        batch.append([v])
      }
      
      names.append(name)
    }
    
    optimizer.isTraining = true

    return names

  }
  
  public func readyUp() async {
    if ready == false || datasetData == nil {
      datasetData = await dataset.build()
      
      if let datasetData {
        compile(dataset: datasetData)
      }
    }
  }
  
  private func compile(dataset: RNNSupportedDatasetData) {
    guard let first = dataset.training.first else { fatalError("Could not build network with dataset") }
    
    let vocabSize = first.data.shape[0]
    let wordLength = first.data.shape[2]
    
    self.vocabSize = vocabSize
    self.wordLength = wordLength
    
    let lstm = LSTM<N>(inputUnits: lstmParameters.inputUnits,
                    batchLength: wordLength,
                    returnSequence: returnSequence,
                    biasEnabled: true,
                    initializer: lstmParameters.lstmInitializer,
                    hiddenUnits: lstmParameters.hiddenUnits,
                    vocabSize: vocabSize)
    
    let embedding = Embedding<N>(inputUnits: lstmParameters.inputUnits,
                              vocabSize: vocabSize,
                              batchLength: wordLength,
                              initializer: lstmParameters.embeddingInitializer,
                              trainable: true)
    
    self.embedding = embedding
    self.lstm = lstm
    
    var layers: [BaseLayer<N>] = [embedding, lstm]
    layers.append(contentsOf: extraLayers)
    
    let sequential = Sequential<N>({ layers })
    
    optimizer.trainable = sequential
    
    ready = true
  }
}

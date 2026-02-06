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
  associatedtype Item: Hashable
  
  var vocabSize: Int { get }
  func oneHot(_ items: [Item]) -> Tensor
  func vectorize(_ items: [Item]) -> Tensor
  func getWord(for data: Tensor, oneHot: Bool) -> [Item]
  func build() async -> RNNSupportedDatasetData
}

public class RNN<Dataset: RNNSupportedDataset>: Classifier where Dataset.Item == String {
  public struct RNNLSTMParameters {
    let hiddenUnits: Int
    let inputUnits: Int
    let embeddingInitializer: InitializerType
    let lstmInitializer: InitializerType

    public init(hiddenUnits: Int,
                inputUnits: Int,
                embeddingInitializer: InitializerType = .xavierUniform,
                lstmInitializer: InitializerType = .xavierUniform) {
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
    let weightDecay: Adam.WeightDecay
    let metricsReporter: MetricsReporter?
    
    public init(learningRate: Tensor.Scalar,
                b1: Tensor.Scalar = 0.9,
                b2: Tensor.Scalar = 0.999,
                eps: Tensor.Scalar = .stabilityFactor,
                weightDecay: Adam.WeightDecay = .none,
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
    let accuracyThreshold: AccuracyThreshold
    let killOnAccuracy: Bool
    let lossFunction: LossFunction
    
    public init(batchSize: Int,
                epochs: Int,
                accuracyThreshold: AccuracyThreshold = .init(value: 0.9, averageCount: 5),
                killOnAccuracy: Bool = true,
                lossFunction: LossFunction = .crossEntropySoftmax) {
      self.batchSize = batchSize
      self.epochs = epochs
      self.accuracyThreshold = accuracyThreshold
      self.killOnAccuracy = killOnAccuracy
      self.lossFunction = lossFunction
    }
  }
  
  private let dataset: Dataset
  private var lstm: LSTM?
  private var embedding: Embedding?
  private var vocabSize: Int = 0
  private var wordLength: Int = 0
  private var extraLayers: [Layer]
  private var ready: Bool = false
  private var datasetData: RNNSupportedDatasetData?
  private let returnSequence: Bool
  
  private let classifierParameters: ClassifierParameters
  private let optimizerParameters: OptimizerParameters
  private let lstmParameters: RNNLSTMParameters
  
  public init(device: Device = CPU(),
              returnSequence: Bool = true,
              dataset: Dataset,
              classifierParameters: ClassifierParameters,
              optimizerParameters: OptimizerParameters,
              lstmParameters: RNNLSTMParameters,
              extraLayers: () -> [Layer] = { [] }) {
    
    self.classifierParameters = classifierParameters
    self.optimizerParameters = optimizerParameters
    self.lstmParameters = lstmParameters
    
    self.returnSequence = returnSequence
    self.dataset = dataset
    self.extraLayers = extraLayers()
    
    let network = Sequential { [] }
    
    let optimizer = Adam(network,
                         device: device,
                         learningRate: optimizerParameters.learningRate,
                         batchSize: classifierParameters.batchSize,
                         b1: optimizerParameters.b1,
                         b2: optimizerParameters.b2,
                         eps: optimizerParameters.eps,
                         weightDecay: optimizerParameters.weightDecay,
                         gradientClip: nil)

    optimizer.metricsReporter = optimizerParameters.metricsReporter
      
    super.init(optimizer: optimizer,
               epochs: classifierParameters.epochs,
               batchSize: classifierParameters.batchSize,
               accuracyThreshold: classifierParameters.accuracyThreshold,
               killOnAccuracy: classifierParameters.killOnAccuracy,
               log: false,
               lossFunction: classifierParameters.lossFunction)
  }
  
  public func importFrom(url: URL?) async {
    guard let url else { return }
    
    await readyUp()
    
    let n = Sequential.import(url)
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
          
      var batchTensor: Tensor
      
      if let with {
        batchTensor = dataset.vectorize([with])
        name += with

      } else {
        let index = Int.random(in: 0..<vocabSize).asTensorScalar
              
        batchTensor = Tensor([[[index]]])
        
        // append random letter
        let unvec = dataset.getWord(for: batchTensor, oneHot: false).joined()
        name += unvec
      }

      while runningChar != endingMark && name.count < maxWordLength {
        
        // still 1 hot encoding
        let out = optimizer.predict([batchTensor])
        
        guard let outTensor = out[safe: 0],
              outTensor.depthSliceCount > 0 else {
          break
        }
        
        // Get the last depth slice (last timestep output)
        let lastDepthIdx = batchTensor.depthSliceCount - 1
        let lastSlice = outTensor.depthSlice(min(lastDepthIdx, outTensor.depthSliceCount - 1))
        let flat = Array(lastSlice)
      
        var v: [Tensor.Scalar] = [Tensor.Scalar](repeating: 0, count: flat.count)
        
        let indexToChoose: Int
        if randomizeSelection {
          indexToChoose = NumSwift.randomChoice(in: Array(0..<vocabSize), p: flat).1
        } else {
          indexToChoose = Int(flat.indexOfMax.0)
        }
        
        v[indexToChoose] = 1
        
        // one hot because we're predicting based on the output which is trained on the labels which are expected to be one-hot encoded
        // TODO: how do we enforce this? 
        let unvec = dataset.getWord(for: Tensor(v), oneHot: true).joined()
        
        runningChar = unvec
        name += unvec
        
        // vectorize again to append to batch using Tensor concat
        let vectorizedLetter = dataset.vectorize([unvec])
        if vectorizedLetter.depthSliceCount > 0 {
          let letterSlice = vectorizedLetter.depthSliceTensor(0)
          batchTensor = batchTensor.concat(letterSlice, axis: 2)
        }
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
    
    let vocabSize = self.dataset.vocabSize
    // average length of the word
    let wordLength = first.data.shape[2] // expect int vectorized array where each value is an index into the vocab size. TODO: enforce this
    
    self.vocabSize = vocabSize
    self.wordLength = wordLength
    
    let lstm = LSTM(inputUnits: lstmParameters.inputUnits,
                    batchLength: wordLength,
                    returnSequence: returnSequence,
                    biasEnabled: true,
                    initializer: lstmParameters.lstmInitializer,
                    hiddenUnits: lstmParameters.hiddenUnits,
                    vocabSize: vocabSize)
    
    let embedding = Embedding(inputUnits: lstmParameters.inputUnits,
                              vocabSize: vocabSize,
                              batchLength: wordLength,
                              initializer: lstmParameters.embeddingInitializer,
                              trainable: true)
    
    self.embedding = embedding
    self.lstm = lstm
    
    var layers: [Layer] = [embedding, lstm]
    layers.append(contentsOf: extraLayers)
    
    let sequential = Sequential({ layers })
    
    optimizer.trainable = sequential
    
    ready = true
  }
}

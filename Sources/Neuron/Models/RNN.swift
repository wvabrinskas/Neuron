//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/28/23.
//

import Foundation
import NumSwift


// Used exclusively for inference on an RNN model
public final class InferenceOnlyRNN: RNN<EmptyRNNDataset> {
  
  /// Creates an inference-only RNN wrapper around an existing network.
  ///
  /// - Parameters:
  ///   - device: Execution device.
  ///   - returnSequence: Whether sequence outputs include all timesteps.
  ///   - vocabSize: Vocabulary size used for decoding predictions.
  ///   - network: Pretrained sequential network.
  public init(device: Device = CPU(),
              returnSequence: Bool = true,
              vocabSize: Int,
              network: Sequential) {
    super.init(device: device,
               returnSequence: returnSequence,
               dataset: EmptyRNNDataset(vocabSize: vocabSize),
               classifierParameters: .init(batchSize: 0, epochs: 0),
               optimizerParameters: .init(learningRate: 0),
               lstmParameters: .init(hiddenUnits: 0, inputUnits: 0))
    
    optimizer.trainable = network
    optimizer.isTraining = false
  }
  
  /// Imports a serialized sequential network from raw data.
  ///
  /// - Parameter data: Serialized model data.
  public override func importFrom(data: Data?) async {
    guard let data else { return }
    let n = Sequential.import(data)
    optimizer.trainable = n
    optimizer.isTraining = false
  }
  
  /// Imports a serialized sequential network from a file URL.
  ///
  /// - Parameter url: URL to serialized model file.
  public override func importFrom(url: URL?) async {
    guard let url else { return }
    let n = Sequential.import(url)
    optimizer.trainable = n
    optimizer.isTraining = false
  }
}


public typealias RNNSupportedDatasetData = (training: [DatasetModel], val: [DatasetModel])
public protocol RNNSupportedDataset {
  associatedtype Item: Hashable
  
  var vocabSize: Int { get }
  /// One-hot encodes dataset items.
  ///
  /// - Parameter items: Items to encode.
  /// - Returns: One-hot tensor representation.
  func oneHot(_ items: [Item]) -> Tensor
  /// Vectorizes dataset items into index-based representation.
  ///
  /// - Parameter items: Items to vectorize.
  /// - Returns: Tensor containing vectorized token IDs.
  func vectorize(_ items: [Item]) -> Tensor
  /// Decodes model output tensor values back into dataset items.
  ///
  /// - Parameters:
  ///   - data: Tensor to decode.
  ///   - oneHot: Whether `data` is one-hot encoded.
  /// - Returns: Decoded item sequence.
  func getWord(for data: Tensor, oneHot: Bool) -> [Item]
  /// Builds training and validation datasets for RNN training.
  ///
  /// - Returns: Tuple containing training and validation datasets.
  func build() async -> RNNSupportedDatasetData
}

public class RNN<Dataset: RNNSupportedDataset>: Classifier where Dataset.Item == String {
  public struct RNNLSTMParameters {
    let hiddenUnits: Int
    let inputUnits: Int
    let embeddingInitializer: InitializerType
    let lstmInitializer: InitializerType

    /// Creates LSTM architecture parameters for an RNN.
    ///
    /// - Parameters:
    ///   - hiddenUnits: Number of hidden LSTM units.
    ///   - inputUnits: Embedding width fed into LSTM.
    ///   - embeddingInitializer: Initializer for embedding weights.
    ///   - lstmInitializer: Initializer for LSTM gate/output weights.
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
    
    /// Creates optimizer hyperparameters for RNN training.
    ///
    /// - Parameters:
    ///   - learningRate: Base learning rate.
    ///   - b1: Adam beta1.
    ///   - b2: Adam beta2.
    ///   - eps: Numerical stability epsilon.
    ///   - weightDecay: Optional Adam weight-decay behavior.
    ///   - metricsReporter: Optional metrics reporter.
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
    
    /// Creates training-loop parameters for `Classifier` behavior.
    ///
    /// - Parameters:
    ///   - batchSize: Batch size used during training.
    ///   - epochs: Number of training epochs.
    ///   - accuracyThreshold: Early-stop threshold policy.
    ///   - killOnAccuracy: Stops training when threshold is reached.
    ///   - lossFunction: Loss function used for optimization.
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
  
  /// Creates an RNN trainer parameterized by a dataset provider.
  ///
  /// - Parameters:
  ///   - device: Execution device.
  ///   - returnSequence: Whether model returns full timestep sequence.
  ///   - dataset: Dataset adapter that vectorizes and builds samples.
  ///   - classifierParameters: Training loop configuration.
  ///   - optimizerParameters: Optimizer hyperparameters.
  ///   - lstmParameters: LSTM architecture hyperparameters.
  ///   - extraLayers: Additional layers appended after embedding/LSTM.
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
  
  /// Imports a serialized network from raw bytes and prepares the trainer.
  ///
  /// - Parameter data: Serialized model data.
  public func importFrom(data: Data?) async {
    guard let data else { return }
    
    await readyUp()
    
    let n = Sequential.import(data)
    optimizer.trainable = n
  }
  
  /// Imports a serialized network from disk and prepares the trainer.
  ///
  /// - Parameter url: URL to serialized model file.
  public func importFrom(url: URL?) async {
    guard let url else { return }
    
    await readyUp()
    
    let n = Sequential.import(url)
    optimizer.trainable = n
  }
  
  /// Builds dataset/network state (if needed) and runs training.
  public func train() async {
    optimizer.isTraining = true
    
    await readyUp()
    
    if let datasetData {
      fit(datasetData.training, datasetData.val)
    }
  }
  
  /// Generates token sequences using iterative autoregressive prediction.
  ///
  /// - Parameters:
  ///   - with: Optional starting token/string prefix.
  ///   - count: Number of sequences to generate.
  ///   - maxWordLength: Maximum generated token count per sequence.
  ///   - randomizeSelection: Samples next token probabilistically when `true`.
  ///   - endingMark: Token that terminates generation.
  /// - Returns: Generated string sequences.
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
              
        batchTensor = Tensor.fillWith(value: index, size: .init(rows: 1, columns: 1, depth: 1))
        
        // append random letter
        let unvec = dataset.getWord(for: batchTensor, oneHot: false).joined()
        name += unvec
      }

      while runningChar != endingMark && name.count < maxWordLength {
        
        // still 1 hot encoding
        let out = optimizer.predict([batchTensor])
        
        guard let outTensor = out[safe: 0],
              outTensor.size.depth > 0 else {
          break
        }
        
        // Get the last depth slice (last timestep output)
        let lastDepthIdx = batchTensor.size.depth - 1
        let lastSlice = outTensor.depthSlice(min(lastDepthIdx, outTensor.size.depth - 1))
        let flat = Array(lastSlice)
      
        var v: Tensor.Value = Tensor.Value(repeating: 0, count: flat.count)
        
        let indexToChoose: Int
        if randomizeSelection {
          indexToChoose = NumSwift.randomChoice(in: Array(0..<vocabSize), p: flat).1
        } else {
          indexToChoose = Int(flat.indexOfMax.0)
        }
        
        v[indexToChoose] = 1
        
        // one hot because we're predicting based on the output which is trained on the labels which are expected to be one-hot encoded
        // TODO: how do we enforce this? 
        let unvec = dataset.getWord(for: Tensor(v, size: .init(rows: 1, columns: flat.count, depth: 1)), oneHot: true).joined()
        
        runningChar = unvec
        name += unvec
        
        // vectorize again to append to batch using Tensor concat
        let vectorizedLetter = dataset.vectorize([unvec])
        if vectorizedLetter.size.depth > 0 {
          let letterSlice = vectorizedLetter.depthSliceTensor(0)
          batchTensor = batchTensor.concat(letterSlice, axis: 2)
        }
      }
      
      names.append(name)
    }
    
    optimizer.isTraining = true

    return names

  }
  
  /// Ensures dataset-derived network state is built and compiled once.
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

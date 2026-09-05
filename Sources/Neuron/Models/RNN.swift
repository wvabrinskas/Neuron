//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/28/23.
//

import Foundation
import NumSwift

/// A recurrent neural network classifier that operates on a vectorizing dataset of `String` items.
public class RNN<Dataset: TokenizingDataset>: Classifier where Dataset.Item == String {
  
  //public typealias Dataset = VectorizingDataset
  
  /// Parameters defining the LSTM architecture used within the RNN.
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
  
  /// Parameters defining the optimizer configuration for RNN training.
  public struct OptimizerParameters {
    let learningRate: Tensor.Scalar
    let b1: Tensor.Scalar
    let b2: Tensor.Scalar
    let eps: Tensor.Scalar
    let weightDecay: Adam.WeightDecay
    let metricsReporter: MetricsReporter?
    let gradientClip: Tensor.Scalar?
    
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
                gradientClip: Tensor.Scalar? = nil,
                metricsReporter: MetricsReporter? = nil) {
      self.learningRate = learningRate
      self.b1 = b1
      self.b2 = b2
      self.eps = eps
      self.metricsReporter = metricsReporter
      self.weightDecay = weightDecay
      self.gradientClip = gradientClip
    }
  }
  
  /// Parameters controlling the training loop behavior of the classifier.
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
                killOnAccuracy: Bool = true) {
      self.batchSize = batchSize
      self.epochs = epochs
      self.accuracyThreshold = accuracyThreshold
      self.killOnAccuracy = killOnAccuracy
      self.lossFunction = .sparseCrossEntropySoftmax
    }
  }
  
  var dataset: Dataset {
    didSet {
      vocabSize = dataset.vocabSize
    }
  }
  private var lstm: LSTM?
  private var embedding: Embedding?
  private var vocabSize: Int = 0
  private var wordLength: Int = 0
  private var extraLayers: [Layer]
  private var ready: Bool = false
  private var datasetData: VectorizingDatasetData?
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
  public init(device: DeviceType = .cpu,
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
    
    self.vocabSize = dataset.vocabSize
    
    let network = Sequential { [] }
    
    let optimizer = Adam(network,
                         learningRate: optimizerParameters.learningRate,
                         batchSize: classifierParameters.batchSize,
                         b1: optimizerParameters.b1,
                         b2: optimizerParameters.b2,
                         eps: optimizerParameters.eps,
                         weightDecay: optimizerParameters.weightDecay,
                         gradientClip: optimizerParameters.gradientClip)
    
    optimizer.deviceType = device
    optimizer.metricsReporter = optimizerParameters.metricsReporter
      
    super.init(optimizer: optimizer,
               epochs: classifierParameters.epochs,
               batchSize: classifierParameters.batchSize,
               accuracyThreshold: classifierParameters.accuracyThreshold,
               killOnAccuracy: classifierParameters.killOnAccuracy,
               log: false,
               lossFunction: classifierParameters.lossFunction)
  }
  
  /// Exports the RNN model along with its embedding vectors to disk.
  ///
  /// - Parameters:
  ///   - overrite: Whether to overwrite an existing file at the destination.
  ///   - compress: Whether to compress the exported files.
  /// - Returns: A tuple containing the URL of the exported model and the URL of the exported vectors, either of which may be `nil` on failure.
  public override func export(overrite: Bool = false, compress: Bool = true) -> URL? {
    fatalError("Please use exportWithVectors")
  }
  
  /// Exports the RNN model along with its associated word vectors.
  ///
  /// - Parameters:
  ///   - overrite: Whether to overwrite existing exported files.
  ///   - compress: Whether to compress the exported output.
  /// - Returns: A tuple containing the optional URL for the exported model and the optional URL for the exported vectors.
  public func exportWithVectors(overrite: Bool = false, compress: Bool = true) -> (model: URL?, vectors: URL?) {
    let model = super.export(overrite: overrite, compress: compress)
    let dataset = dataset.export(name: "vectors", overrite: overrite, compress: compress)
    return (model, dataset)
  }
  
  /// Imports a serialized network from raw bytes and prepares the trainer.
  ///
  /// - Parameters:
  ///   - data: Serialized model bytes for the Sequential network.
  ///   - vectors: Serialized vectorizer data used to rebuild the dataset vocabulary.
  public func importFrom(data: Data?, vectors: Data?) async {
    guard let data, let vectors else { return }
    
    dataset = Dataset.build(data: vectors)
    
    await readyUp()
      
    let n = Sequential.import(data)
    optimizer.trainable = n
  }
  
  /// Imports a serialized network from disk and prepares the trainer.
  ///
  /// - Parameters:
  ///   - url: File URL pointing to the serialized Sequential network.
  ///   - vectors: File URL pointing to the serialized vectorizer for the dataset vocabulary.
  public func importFrom(url: URL?, vectors: URL?) async {
    guard let url, let vectors else { return }
    
    dataset = Dataset.build(url: vectors)
    
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
                      maxTokenCount: Int = 20,
                      randomizeSelection: Bool = false,
                      delimiter: String = "",
                      endingMark: String = ".") -> [String] {
    optimizer.isTraining = false
    
    var names: [String] = []
    
    for _ in 0..<count {
      
      // Collect IDs and decode once at the end. Decoding token by token and concatenating
      // loses every word boundary: `decode` turns `</w>` into a space and then trims it, so
      // "the cat sat" comes back as "thecatsat". Only a whole-sequence decode sees the
      // boundaries between tokens.
      var tokenIds: [Int] = []
      var finished: Bool = false
          
      var batchTensor: Tensor
      
      if let with {
        batchTensor = dataset.tokenize(with)
        tokenIds.append(contentsOf: batchTensor.storage.map { Int($0) })

      } else {
        // Seed from a token that actually renders. Control tokens are in the ID range but
        // decode to nothing, which would silently produce an empty sequence start.
        let seedIds = (0..<vocabSize).filter { dataset.controlTokenIds.contains($0) == false }
        let seed = seedIds.randomElement() ?? 0
              
        batchTensor = Tensor(seed.asTensorScalar)
        
        tokenIds.append(seed)
      }
      
      // we use tokenCount instead of `name.count` because we want to account for sentence structure as well
      // just using count would result in sentence truncation.
      var currentTokenCount = 1

      while finished == false && currentTokenCount < maxTokenCount {
        
        let out = optimizer.predict([batchTensor])
        
        guard let outTensor = out[safe: 0],
              outTensor.size.depth > 0 else {
          break
        }
        
        // Get the last depth slice (last timestep output). The slice is a distribution over the
        // vocabulary, so the next token is its argmax (or a probabilistic draw) -- NOT its first
        // element, which is just P(token 0).
        let lastDepthIdx = batchTensor.size.depth - 1
        let lastSlice = Array(outTensor.depthSlice(min(lastDepthIdx, outTensor.size.depth - 1)))

        guard lastSlice.isEmpty == false else { break }

        let tokenId: Int
        if randomizeSelection {
          tokenId = NumSwift.randomChoice(in: Array(0..<lastSlice.count), p: lastSlice).1
        } else {
          tokenId = Int(lastSlice.indexOfMax.0)
        }

        // Stop on the tokenizer's own end-of-sequence ID. Comparing decoded text can't do this
        // reliably: decoding is lossy (`</w>` becomes a space and is then trimmed away, so a
        // terminal token can render as ""), and BPE merges the ending mark into a larger token,
        // so "ley." arrives as one token that never *equals* ".".
        finished = tokenId == dataset.eosTokenId
        
        if finished == false {
          tokenIds.append(tokenId)
          
          if endingMark.isEmpty == false,
             dataset.item(for: Tensor(Tensor.Scalar(tokenId))).contains(endingMark) {
            finished = true
          }
        }
        
        // Append the predicted ID itself. Re-encoding the *decoded* text would push it back
        // through BPE, which can split one token into several and desync the sequence.
        batchTensor = batchTensor.concat(Tensor([[[Tensor.Scalar(tokenId)]]]), axis: 2)
        
        currentTokenCount += 1
      }
      
      names.append(assemble(tokenIds: tokenIds, delimiter: delimiter))
    }
    
    optimizer.isTraining = true

    return names

  }
  
  /// Renders generated token IDs as text.
  ///
  /// - Parameters:
  ///   - tokenIds: IDs produced by generation, in order.
  ///   - delimiter: Inserted between tokens. When empty the whole sequence is decoded in one
  ///     pass so word boundaries survive; a non-empty delimiter needs per-token strings, and
  ///     the caller has opted into that separator carrying the boundary instead.
  /// - Returns: The generated text.
  private func assemble(tokenIds: [Int], delimiter: String) -> String {
    guard delimiter.isEmpty == false else {
      return dataset.item(for: Tensor(tokenIds.map { [[Tensor.Scalar($0)]] }))
    }

    return tokenIds
      .map { dataset.item(for: Tensor(Tensor.Scalar($0))) }
      .filter { $0.isEmpty == false }
      .joined(separator: delimiter)
  }
  
  /// Ensures dataset-derived network state is built and compiled once.
  public func readyUp() async {
    // do it twice just incase we imported a dataset and there's no data to build
    vocabSize = dataset.vocabSize

    if ready == false || datasetData == nil {
      datasetData = await dataset.build()
      
      vocabSize = dataset.vocabSize

      // Sequences are padded to a fixed length, so most batches carry timesteps whose only
      // correct answer is <pad>. Scoring those teaches the model nothing and reports an
      // accuracy dominated by how much padding the batch happened to need.
      optimizer.ignoreLabelIndex = dataset.padTokenId

      if let datasetData {
        compile(dataset: datasetData)
      }
    }
  }
  
  /// Verifies every sample carries the sequence length the network is about to compile against.
  ///
  /// - Parameters:
  ///   - dataset: Training and validation samples to check.
  ///   - wordLength: Token count taken from the first training sample.
  private func validate(dataset: VectorizingDatasetData, wordLength: Int) {
    // With `returnSequence` the model emits one distribution per timestep and the sparse loss
    // wants an index for each; otherwise it emits only the final one.
    let expectedLabelCount = returnSequence ? wordLength : 1
    
    for (name, samples) in [("training", dataset.training), ("validation", dataset.val)] {
      for (index, sample) in samples.enumerated() {
        guard sample.data.size.depth == wordLength else {
          fatalError("RNN requires every sample to carry the same token count. \(name) sample \(index) has depth \(sample.data.size.depth), expected \(wordLength). Pad sequences to a fixed length when building the dataset -- see `tokenize(_:paddedTo:appendingEnd:)`.")
        }
        
        guard sample.label.storage.count >= expectedLabelCount else {
          fatalError("RNN requires at least \(expectedLabelCount) label indices per sample, one per predicted timestep. \(name) sample \(index) has \(sample.label.storage.count).")
        }
      }
    }
  }
  
  private func compile(dataset: VectorizingDatasetData) {
    guard let first = dataset.training.first else {
      print("Could not build network with dataset")
      return
    }
    
    // Sequence length in tokens. Every sample has to agree on it: `LSTM.forward` iterates a
    // fixed `0..<batchLength`, zero-filling timesteps a short sample doesn't provide and never
    // reading the ones a long sample carries past the window. Neither is reported, so a ragged
    // dataset trains on silently corrupted sequences -- and BPE makes datasets ragged by
    // default, since equal character counts no longer mean equal token counts.
    let wordLength = first.data.shape[2]
    
    validate(dataset: dataset, wordLength: wordLength)
    
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

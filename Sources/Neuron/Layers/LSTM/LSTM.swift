//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/2/23.
//

import Foundation
import NumSwift

/// The LSTM layer, Long Short-Term Memory layer, is the heart of the `RNN` model. It should be
/// preceeded by an `Embedding` layer as that's the expected input rather than the raw
/// text input itself.
public final class LSTM: Layer {
  public var encodingType: EncodingType = .lstm
  public var inputSize: TensorSize = TensorSize(array: [0,0,0])
  public var outputSize: TensorSize = TensorSize(array: [0,0,0])
  public var weights: Tensor = Tensor()
  public var biases: Tensor = Tensor()
  public var biasEnabled: Bool = false
  public var trainable: Bool = true
  public var initializer: Initializer?
  public var device: Device = CPU()
  public var isTraining: Bool = true

  public var forgetGateWeights: Tensor = Tensor()
  public var inputGateWeights: Tensor = Tensor()
  public var gateGateWeights: Tensor = Tensor()
  public var outputGateWeights: Tensor = Tensor()
  public var hiddenOutputWeights: Tensor = Tensor()
    
  private var hiddenUnits: Int
  private var vocabSize: Int
  private var inputUnits: Int
  private var batchLength: Int
  private let returnSequence: Bool

  public class LSTMActivations {
    let forgetGate: Tensor
    let inputGate: Tensor
    let outputGate: Tensor
    let gateGate: Tensor
    
    init(activations: LSTMCell.Activations) {
      self.forgetGate = activations.fa
      self.inputGate = activations.ia
      self.outputGate = activations.oa
      self.gateGate = activations.ga
    }
    
    init(forgetGate: Tensor = .init(),
         inputGate: Tensor = .init(),
         outputGate: Tensor = .init(),
         gateGate: Tensor = .init()) {
      self.forgetGate = forgetGate
      self.inputGate = inputGate
      self.outputGate = outputGate
      self.gateGate = gateGate
    }
  }
  
  public class Cache {
    var lstm: LSTMActivations
    var cell: Tensor
    var activation: Tensor
    var embedding: Tensor
    var output: Tensor
    
    init(lstm: LSTMActivations = .init(),
         cell: Tensor = .init(),
         activation: Tensor = .init(),
         embedding: Tensor = .init(),
         output: Tensor = .init()) {
      self.lstm = lstm
      self.cell = cell
      self.activation = activation
      self.embedding = embedding
      self.output = output
    }
    
    func updating(lstm: LSTMActivations? = nil,
                  cell: Tensor? = nil,
                  activation: Tensor? = nil,
                  embedding: Tensor? = nil,
                  output: Tensor? = nil) {
      self.lstm = lstm ?? self.lstm
      self.cell = cell ?? self.cell
      self.activation = activation ?? self.activation
      self.embedding = embedding ?? self.embedding
      self.output = embedding ?? self.output
    }
  }

  
  /// Default initializer
  /// - Parameters:
  ///   - inputUnits: The number of inputs in the LSTM cell
  ///   - batchLength: The number samples (eg. letters) at a given time
  ///   - returnSequence: Determines if the layer returns all outputs of the sequence or just the last output. Default:   `true`
  ///   - initializer: Initializer funciton to use
  ///   - hiddenUnits: Number of hidden use
  ///   - vocabSize: size of the expected vocabulary
  public init(inputUnits: Int,
              batchLength: Int,
              returnSequence: Bool = true,
              initializer: InitializerType = .xavierNormal,
              hiddenUnits: Int,
              vocabSize: Int) {
    self.inputSize = TensorSize(rows: 1,
                                columns: vocabSize,
                                depth: batchLength)
    self.initializer = initializer.build()
    self.hiddenUnits = hiddenUnits
    self.vocabSize = vocabSize
    self.inputUnits = inputUnits
    self.batchLength = batchLength
    self.outputSize = TensorSize(rows: 1,
                                 columns: vocabSize,
                                 depth: returnSequence ? batchLength : 1)
    
    self.returnSequence = returnSequence
        
    initializeWeights()
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         biasEnabled,
         outputSize,
         biases,
         type,
         forgetGateWeights,
         inputGateWeights,
         gateGateWeights,
         outputGateWeights,
         hiddenUnits,
         vocabSize,
         hiddenOutputWeights,
         batchLength,
         inputUnits
  }
  
  convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let hiddenUnits = try container.decodeIfPresent(Int.self, forKey: .hiddenUnits) ?? 0
    let vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 0
    let inputUnits = try container.decodeIfPresent(Int.self, forKey: .inputUnits) ?? 0
    let batchLength = try container.decodeIfPresent(Int.self, forKey: .batchLength) ?? 0

    self.init(inputUnits: inputUnits,
              batchLength: batchLength,
              hiddenUnits: hiddenUnits,
              vocabSize: vocabSize)

    self.biases = try container.decodeIfPresent(Tensor.self, forKey: .biases) ?? Tensor()
    self.biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    self.outputSize = try container.decodeIfPresent(TensorSize.self, forKey: .outputSize) ?? TensorSize(array: [])
    self.forgetGateWeights = try container.decodeIfPresent(Tensor.self, forKey: .forgetGateWeights) ?? Tensor()
    self.inputGateWeights = try container.decodeIfPresent(Tensor.self, forKey: .inputGateWeights) ?? Tensor()
    self.gateGateWeights = try container.decodeIfPresent(Tensor.self, forKey: .gateGateWeights) ?? Tensor()
    self.outputGateWeights = try container.decodeIfPresent(Tensor.self, forKey: .outputGateWeights) ?? Tensor()
    self.hiddenOutputWeights = try container.decodeIfPresent(Tensor.self, forKey: .hiddenOutputWeights) ?? Tensor()
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(biases, forKey: .biases)
    try container.encode(outputSize, forKey: .outputSize)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(biasEnabled, forKey: .biasEnabled)
    try container.encode(forgetGateWeights, forKey: .forgetGateWeights)
    try container.encode(inputGateWeights, forKey: .inputGateWeights)
    try container.encode(gateGateWeights, forKey: .gateGateWeights)
    try container.encode(outputGateWeights, forKey: .outputGateWeights)
    try container.encode(hiddenOutputWeights, forKey: .hiddenOutputWeights)
    try container.encode(hiddenUnits, forKey: .hiddenUnits)
    try container.encode(vocabSize, forKey: .vocabSize)
    try container.encode(batchLength, forKey: .batchLength)
    try container.encode(inputUnits, forKey: .inputUnits)
  }
  
  
  /// The forward path for the LSTM layer. Should be preceeded by an `Embedding` layer.
  /// Emdedding input size expected is `(rows: 1, columns: inputUnits, depth: batchSize)`
  /// - Parameter tensor: The `Embedding` input `Tensor`
  /// - Returns: Depending on the state of `returnSequence` it will either returng the whole sequence of size
  /// `(rows: 1, columns: vocabSize, depth: batchSize)` or just the last output of the sequence of size
  /// `(rows: 1, columns: vocabSize, depth: 1)`
  public func forward(tensor: Tensor) -> Tensor {
    
    var cellCache: [Cache] = [setupInitialState()]
    var cells: [(LSTMCell, OutputCell)] = []

    let context = TensorContext { inputs, gradient in
      var eat: [[Tensor.Scalar]] = NumSwift.zerosLike((rows: 1,
                                                       columns: self.hiddenUnits))
      var ect: [[Tensor.Scalar]] = eat
      
      var wrtOutputWeightsDerivatives: Tensor = Tensor()
      var wrtLSTMCellInputWeightsDerivatives: LSTMCell.ParameterDerivatives = .init()
      
      var wrtEmbeddings: Tensor = Tensor()
      
      for i in 0..<cellCache.count {
        // so we dont have to reverse the array
        let index = (cellCache.count - 1) - i

        let cache = cellCache[index]
        let previousCache = cellCache[safe: index - 1]
        
        let outputCell = cells[index].1

        let activationErrors = outputCell.backward(gradient: gradient.value[safe: index] ?? gradient.zerosLike().value[0], // if we dont emit the sequence then they dont effect the output so we zero them
                                                   activations: cellCache[index].activation.value[0], // should be depth of 1 always
                                                   batchSize: self.batchLength,
                                                   hiddenOutputWeights: self.hiddenOutputWeights)
        
        if wrtOutputWeightsDerivatives.isEmpty {
          wrtOutputWeightsDerivatives = activationErrors.weights
        } else {
          wrtOutputWeightsDerivatives = wrtOutputWeightsDerivatives + activationErrors.weights
        }
        
        let nextActivationError = eat
        let activationOutputError = activationErrors.outputs.value[0]

        let cell = cells[index]
        let backward = cell.0.backward(cache: cache,
                                       previousCache: previousCache,
                                       activationOutputError: activationOutputError,
                                       nextActivationError: nextActivationError,
                                       nextCellError: ect,
                                       batchSize: self.batchLength,
                                       parameters: .init(forgetGateWeights: self.forgetGateWeights,
                                                         inputGateWeights: self.inputGateWeights,
                                                         gateGateWeights: self.gateGateWeights,
                                                         outputGateWeights: self.outputGateWeights))
        
        if wrtLSTMCellInputWeightsDerivatives.isEmpty {
          wrtLSTMCellInputWeightsDerivatives = backward.weights
        } else {
          wrtLSTMCellInputWeightsDerivatives = wrtLSTMCellInputWeightsDerivatives + backward.weights
        }
        
        let previousCellError = backward.inputs.previousCellError
        let previousActivationError = backward.inputs.previousActivationError
        
        let embeddingError = backward.inputs.embeddingError
        
        if wrtEmbeddings.isEmpty {
          wrtEmbeddings = embeddingError
        } else {
          let dEmbed = wrtEmbeddings.concat(embeddingError, axis: 2)
          wrtEmbeddings = dEmbed
        }
        
        if let pae = previousActivationError.value[safe: 0],
           let pce = previousCellError.value[safe: 0] {
          eat = pae
          ect = pce
        }
      }
      
      // merge all weights into a giant 5 depth tensor, shape will be broken here
      let weightDerivatives = wrtLSTMCellInputWeightsDerivatives.concat().concat(wrtOutputWeightsDerivatives, axis: 2)
      
      return (wrtEmbeddings, weightDerivatives)
    }
    
    var out = Tensor(context: context)
    
    // TODO: figure out how to avoid setting batch length and have it be dynamic.
    // Right now we are just using a hard list of batchLength and generating that length
    for d in 0..<batchLength {
      guard let cache = cellCache[safe: d] else { break }

      // get embeddings from input
      let getEmbeddings = Tensor(tensor.value[safe: d] ?? out.value[safe: d] ?? tensor.value[0])
      
      let cell = cells[safe: d]?.0 ?? LSTMCell(hidden: hiddenUnits,
                                               input: inputUnits,
                                               vocabSize: vocabSize,
                                               batchSize: batchLength)
                  
      let cellParameters = LSTMCell.Parameters(forgetGateWeights: forgetGateWeights.detached(),
                                               inputGateWeights: inputGateWeights.detached(),
                                               gateGateWeights: gateGateWeights.detached(),
                                               outputGateWeights: outputGateWeights.detached())

      let cellOutput = cell.forward(tensor: getEmbeddings,
                                    parameters: cellParameters,
                                    cache: cache)
      
      let outputCell = cells[safe: d]?.1 ?? OutputCell(device: device)
      let outputCellParameters = OutputCell.Parameters(hiddenOutputWeights: hiddenOutputWeights.detached(),
                                                       activationMatrix: cellOutput.activationMatrix.detached())
      
      let outputCellOutput = outputCell.forward(parameters: outputCellParameters)
          
      let newCellCache = Cache(lstm: LSTMActivations(activations: cellOutput),
                               cell: cellOutput.cellMemoryMatrix.detached(),
                               activation: cellOutput.activationMatrix.detached(),
                               embedding: getEmbeddings.detached(),
                               output: outputCellOutput.detached())
      
      if cellCache[safe: d + 1] != nil {
        cellCache[d + 1] = newCellCache
      } else {
        cellCache.append(newCellCache)
      }
      
      if cells[safe: d + 1] != nil {
        cells[d + 1] = (cell, outputCell)
      } else {
        cells.append((cell, outputCell))
      }
      
      let new = out.concat(outputCellOutput, axis: 2)
      out = new
    }
    
    // drop first state since it's just default values
    cellCache = Array(cellCache.dropFirst())

    if returnSequence == false, let last = out.value.last {
      out = Tensor(last, context: context)
    }
    
    out.label = String(describing: self)
    out.setGraph(tensor)
    
    return out
  }
  
  public func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Float) {
    /*
     order of weights in tensor...
     
     dForgetGateWeights = 0
     dInputGateWeights = 1
     dGateGateWeights = 2
     dOutputGateWeights = 3
     
     hiddenOutputWeightGradients = 4
     */

    var gLayers = gradients.weights.value.reshape(columns: 1)
    
    guard let forgetGateWeightGrads = gLayers[safe: 0],
          let inputGateWeightGrads = gLayers[safe: 1],
          let gateGateWeightGrads = gLayers[safe: 2],
          let outputGateWeightGrads = gLayers[safe: 3],
          let hiddenOutputWeightGradients = gLayers[safe: 4]?[..<vocabSize, 0..<hiddenUnits, 0...] else {
      fatalError("Certain gate weights are not in the gradient")
    }

    gLayers = gLayers.dropLast()

    self.forgetGateWeights = self.forgetGateWeights - Tensor(forgetGateWeightGrads)
    self.inputGateWeights = self.inputGateWeights - Tensor(inputGateWeightGrads)
    self.gateGateWeights = self.gateGateWeights - Tensor(gateGateWeightGrads)
    self.outputGateWeights = self.outputGateWeights - Tensor(outputGateWeightGrads)

    self.hiddenOutputWeights = self.hiddenOutputWeights - Tensor(hiddenOutputWeightGradients)
  }
  
  
  // MARK: Private
  private func initializeWeights() {
    guard let initializer = self.initializer else { return }
        
    let totalInputSize = inputUnits + hiddenUnits
    let weightSize = TensorSize(rows: totalInputSize,
                                columns: hiddenUnits,
                                depth: 1)
    
    let forgetWeights = initializer.calculate(size: weightSize,
                                              input: inputUnits * vocabSize,
                                              out: inputUnits * vocabSize)
    
    let inputWeights = initializer.calculate(size: weightSize,
                                             input: inputUnits * vocabSize,
                                             out: inputUnits * vocabSize)
    
    let gateWeights = initializer.calculate(size: weightSize,
                                            input: inputUnits * vocabSize,
                                            out: inputUnits * vocabSize)
    
    let outputGateWeights = initializer.calculate(size: weightSize,
                                                  input: inputUnits * vocabSize,
                                                  out: inputUnits * vocabSize)
    
    let outputWeights = initializer.calculate(size: TensorSize(array: [vocabSize, hiddenUnits, 1]),
                                              input: inputUnits * vocabSize,
                                              out: inputUnits * vocabSize)
    
    
    self.outputGateWeights = outputGateWeights
    self.forgetGateWeights = forgetWeights
    self.gateGateWeights = gateWeights
    self.inputGateWeights = inputWeights
    self.hiddenOutputWeights = outputWeights
  }
  
  private func setupInitialState() -> Cache {
    let a = Tensor(NumSwift.zerosLike((rows: batchLength, columns: hiddenUnits, depth: 1)))
    let c = Tensor(NumSwift.zerosLike((rows: batchLength, columns: hiddenUnits, depth: 1)))
    
    let initialCache = Cache()
    
    initialCache.activation = a
    initialCache.cell = c

    return initialCache
  }

}

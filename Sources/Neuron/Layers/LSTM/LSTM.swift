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
public final class LSTM: BaseLayer {
  public override var isTraining: Bool {
    willSet {
      if newValue == false {
        reset()
      }
    }
  }

  public var forgetGateWeights: Tensor = Tensor()
  public var forgetGateBiases: Tensor = Tensor()

  public var inputGateWeights: Tensor = Tensor()
  public var inputGateBiases: Tensor = Tensor()

  public var gateGateWeights: Tensor = Tensor()
  public var gateGateBiases: Tensor = Tensor()

  public var outputGateWeights: Tensor = Tensor()
  public var outputGateBiases: Tensor = Tensor()

  public var hiddenOutputWeights: Tensor = Tensor()
  public var hiddenOutputBiases: Tensor = Tensor()
    
  private var hiddenUnits: Int
  private var vocabSize: Int
  private var inputUnits: Int
  private var batchLength: Int
  private let returnSequence: Bool
  
  private var cellCache: ThreadStorage<[Cache]> = .init()

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
    var output: OutputCell?
    var outputValue: Tensor
    
    init(lstm: LSTMActivations = .init(),
         cell: Tensor = .init(),
         activation: Tensor = .init(),
         embedding: Tensor = .init(),
         output: OutputCell? = nil,
         outputValue: Tensor = .init()) {
      self.lstm = lstm
      self.cell = cell
      self.activation = activation
      self.embedding = embedding
      self.output = output
      self.outputValue = outputValue
    }
    
    func updating(lstm: LSTMActivations? = nil,
                  cell: Tensor? = nil,
                  activation: Tensor? = nil,
                  embedding: Tensor? = nil,
                  output: OutputCell? = nil,
                  outputValue: Tensor? = nil) {
      self.lstm = lstm ?? self.lstm
      self.cell = cell ?? self.cell
      self.activation = activation ?? self.activation
      self.embedding = embedding ?? self.embedding
      self.output = output ?? self.output
      self.outputValue = outputValue ?? self.outputValue
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
              biasEnabled: Bool = false,
              initializer: InitializerType = .xavierNormal,
              hiddenUnits: Int,
              vocabSize: Int) {
    let inputSize = TensorSize(rows: 1,
                                columns: vocabSize,
                                depth: batchLength)
    self.hiddenUnits = hiddenUnits
    self.vocabSize = vocabSize
    self.inputUnits = inputUnits
    self.batchLength = batchLength
    self.returnSequence = returnSequence
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
               encodingType: .lstm)
    
    initializeWeights()
    if biasEnabled {
      initializeBiases()
    }
  }
  
  enum CodingKeys: String, CodingKey {
    case inputSize,
         biasEnabled,
         outputSize,
         type,
         forgetGateWeights,
         inputGateWeights,
         gateGateWeights,
         outputGateWeights,
         forgetGateBiases,
         inputGateBiases,
         gateGateBiases,
         outputGateBiases,
         hiddenUnits,
         vocabSize,
         hiddenOutputWeights,
         hiddenOutputBiases,
         batchLength,
         inputUnits
  }
  
  convenience required public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let hiddenUnits = try container.decodeIfPresent(Int.self, forKey: .hiddenUnits) ?? 0
    let vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 0
    let inputUnits = try container.decodeIfPresent(Int.self, forKey: .inputUnits) ?? 0
    let batchLength = try container.decodeIfPresent(Int.self, forKey: .batchLength) ?? 0

    self.init(inputUnits: inputUnits,
              batchLength: batchLength,
              hiddenUnits: hiddenUnits,
              vocabSize: vocabSize)

    self.biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    self.outputSize = try container.decodeIfPresent(TensorSize.self, forKey: .outputSize) ?? TensorSize(array: [])
    self.forgetGateWeights = try container.decodeIfPresent(Tensor.self, forKey: .forgetGateWeights) ?? Tensor()
    self.inputGateWeights = try container.decodeIfPresent(Tensor.self, forKey: .inputGateWeights) ?? Tensor()
    self.gateGateWeights = try container.decodeIfPresent(Tensor.self, forKey: .gateGateWeights) ?? Tensor()
    self.outputGateWeights = try container.decodeIfPresent(Tensor.self, forKey: .outputGateWeights) ?? Tensor()
    self.hiddenOutputWeights = try container.decodeIfPresent(Tensor.self, forKey: .hiddenOutputWeights) ?? Tensor()
    self.forgetGateBiases = try container.decodeIfPresent(Tensor.self, forKey: .forgetGateBiases) ?? Tensor()
    self.inputGateBiases = try container.decodeIfPresent(Tensor.self, forKey: .inputGateBiases) ?? Tensor()
    self.gateGateBiases = try container.decodeIfPresent(Tensor.self, forKey: .gateGateBiases) ?? Tensor()
    self.outputGateBiases = try container.decodeIfPresent(Tensor.self, forKey: .outputGateBiases) ?? Tensor()
    self.hiddenOutputBiases = try container.decodeIfPresent(Tensor.self, forKey: .hiddenOutputBiases) ?? Tensor()
    
    if forgetGateBiases.isEmpty ||
        inputGateBiases.isEmpty ||
        gateGateBiases.isEmpty ||
        outputGateBiases.isEmpty ||
        hiddenOutputBiases.isEmpty {
      initializeBiases()
    }
    
    if forgetGateWeights.isEmpty ||
        inputGateWeights.isEmpty ||
        gateGateWeights.isEmpty ||
        outputGateWeights.isEmpty ||
        hiddenOutputWeights.isEmpty {
      initializeWeights()
    }
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
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
    try container.encode(forgetGateBiases, forKey: .forgetGateBiases)
    try container.encode(inputGateBiases, forKey: .inputGateBiases)
    try container.encode(gateGateBiases, forKey: .gateGateBiases)
    try container.encode(outputGateBiases, forKey: .outputGateBiases)
    try container.encode(hiddenOutputBiases, forKey: .hiddenOutputBiases)
  }
  
  
  /// The forward path for the LSTM layer. Should be preceeded by an `Embedding` layer.
  /// Emdedding input size expected is `(rows: 1, columns: inputUnits, depth: batchLength)`
  /// - Parameter tensor: The `Embedding` input `Tensor`
  /// - Returns: Depending on the state of `returnSequence` it will either returng the whole sequence of size
  /// `(rows: 1, columns: vocabSize, depth: batchLength)` or just the last output of the sequence of size
  /// `(rows: 1, columns: vocabSize, depth: 1)`
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    var localCellCache: [Cache] = [setupInitialState()]
    
    let tensorContext = TensorContext { self.backward(inputs: $0, gradient: $1, cellCache: localCellCache) }
    
    var out = Tensor(context: tensorContext)

    let range = 0..<batchLength
        
    /// What happens to the prediction after we extend pass the batchLength?
    /// we need to truncate the input data if this happens to fit the expected window length
    for index in range {
      guard let cache = localCellCache[safe: index] else { break }
      
      // get embeddings from input
      let getEmbeddings = Tensor(tensor.value[safe: index] ?? NumSwift.zerosLike((rows: 1, columns: vocabSize))) //use first vector
      
      let cell = LSTMCell(hidden: hiddenUnits,
                          input: inputUnits,
                          vocabSize: vocabSize)
      
      let cellParameters = LSTMCell.Parameters(forgetGateWeights: forgetGateWeights.detached(),
                                               inputGateWeights: inputGateWeights.detached(),
                                               gateGateWeights: gateGateWeights.detached(),
                                               outputGateWeights: outputGateWeights.detached(),
                                               forgetGateBiases: forgetGateBiases.detached(),
                                               inputGateBiases: inputGateBiases.detached(),
                                               gateGateBiases: gateGateBiases.detached(),
                                               outputGateBiases: outputGateBiases.detached())

      let cellOutput = cell.forward(tensor: getEmbeddings,
                                    context: context,
                                    parameters: cellParameters,
                                    cache: cache)
      
      // used mainly for prediction and shouldn't be used in back propogation unless there's a gradient associated with it
      
      let outputCellParameters = OutputCell.Parameters(hiddenOutputWeights: hiddenOutputWeights.detached(),
                                                       hiddenOutputBiases: hiddenOutputBiases.detached(),
                                                       activationMatrix: cellOutput.activationMatrix.detached(),
                                                       vocabSize: vocabSize,
                                                       hiddenSize: hiddenUnits)
      
      let outputCell = cache.output ?? OutputCell(device: device, parameters: outputCellParameters)

      
      // TODO: Figure out what to do with this. we might have to store this as well in the cache.
      let outputCellOutput = outputCell.forward(parameters: outputCellParameters)
          
      let newCellCache = Cache(lstm: LSTMActivations(activations: cellOutput),
                               cell: cellOutput.cellMemoryMatrix.detached(),
                               activation: cellOutput.activationMatrix.detached(),
                               embedding: getEmbeddings.detached(),
                               output: outputCell,
                               outputValue: outputCellOutput) // don't detach we'll use for backprop
      
      localCellCache.append(newCellCache)
      
      let new = out.concat(outputCellOutput, axis: 2)
      out = new
    }
    
    self.cellCache.store(localCellCache, at: context.threadId)

    if returnSequence == false, let last = out.value.last {
      out = Tensor(last, context: tensorContext)
    }
    
    out.label = String(describing: self)
    out.setGraph(tensor)
    
    return out
  }
  
  
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    /*
     order of weights in tensor...
     
     dForgetGateWeights = 0
     dInputGateWeights = 1
     dGateGateWeights = 2
     dOutputGateWeights = 3
     
     hiddenOutputWeightGradients = 4
     */

    var gLayers = gradients.weights.value.reshape(columns: 1)
    
    if let forgetGateWeightGrads = gLayers[safe: 0],
       let inputGateWeightGrads = gLayers[safe: 1],
       let gateGateWeightGrads = gLayers[safe: 2],
       let outputGateWeightGrads = gLayers[safe: 3],
       let hiddenOutputWeightGradients = gLayers[safe: 4]?[..<hiddenUnits, 0..<vocabSize, 0...]  {
      
      gLayers = gLayers.dropLast()
      
      self.forgetGateWeights = self.forgetGateWeights - Tensor(forgetGateWeightGrads)
      self.inputGateWeights = self.inputGateWeights - Tensor(inputGateWeightGrads)
      self.gateGateWeights = self.gateGateWeights - Tensor(gateGateWeightGrads)
      self.outputGateWeights = self.outputGateWeights - Tensor(outputGateWeightGrads)
      
      self.hiddenOutputWeights = self.hiddenOutputWeights - Tensor(hiddenOutputWeightGradients)
    }
    
    /*
     order of biases in tensor...
     
     dForgetGateBiases = 0
     dInputGateBiases = 1
     dGateGateBiases = 2
     dOutputGateBiases = 3
     
     hiddenOutputWeightBiases = 4
     */
    let gBiasLayers = gradients.biases.value.flatten()

    if biasEnabled,
       let forgetGateBiasGrads = gBiasLayers[safe: 0],
       let inputGateBiasGrads = gBiasLayers[safe: 1],
       let gateGateBiasGrads = gBiasLayers[safe: 2],
       let outputGateBiasGrads = gBiasLayers[safe: 3],
       let hiddenOutputBiasGradients = gBiasLayers[safe: 4] {
      
      forgetGateBiases = forgetGateBiases - Tensor(forgetGateBiasGrads)
      inputGateBiases = inputGateBiases - Tensor(inputGateBiasGrads)
      gateGateBiases = gateGateBiases - Tensor(gateGateBiasGrads)
      outputGateBiases = outputGateBiases - Tensor(outputGateBiasGrads)
      hiddenOutputBiases = hiddenOutputBiases - Tensor(hiddenOutputBiasGradients)
    }

    reset()
  }
  
  // MARK: Private
  private func backward(inputs: Tensor, gradient: Tensor, cellCache: [Cache]) -> TensorContext.TensorBackpropResult {
    var eat: [[Tensor.Scalar]] = NumSwift.zerosLike((rows: 1,
                                                     columns: self.hiddenUnits))
    var ect: [[Tensor.Scalar]] = eat
    
    var wrtOutputWeightsDerivatives: Tensor = Tensor()
    var wrtOutputBiasesDerivatives: Tensor = Tensor()
    var wrtLSTMCellInputWeightsDerivatives: LSTMCell.ParameterDerivatives = .init()
    var wrtLSTMCellInputBiasDerivatives: LSTMCell.ParameterDerivatives = .init()

    var wrtEmbeddings: Tensor = Tensor()
          
    for index in (1..<cellCache.count).reversed() {
      
      let cache = cellCache[index]
      let previousCache = cellCache[safe: index - 1]
            
      let activationErrors = cache.outputValue.gradients(delta: Tensor(gradient.value[safe: index] ?? gradient.zerosLike().value[0]))
      
      if wrtOutputWeightsDerivatives.isEmpty {
        wrtOutputWeightsDerivatives = activationErrors.weights[safe: 0, Tensor()]
      } else {
        wrtOutputWeightsDerivatives = wrtOutputWeightsDerivatives + activationErrors.weights[safe: 0, Tensor()]
      }
      
      if wrtOutputBiasesDerivatives.isEmpty {
        wrtOutputBiasesDerivatives = activationErrors.biases[safe: 0, Tensor()]
      } else {
        wrtOutputBiasesDerivatives = wrtOutputBiasesDerivatives + activationErrors.biases[safe: 0, Tensor()]
      }
      
      let nextActivationError = eat
      let activationOutputError = activationErrors.input[safe: 0, Tensor()].value[0]

      let cell = LSTMCell(hidden: self.hiddenUnits,
                          input: self.inputUnits,
                          vocabSize: self.vocabSize,
                          device: self.device)
      
      let backward = cell.backward(cache: cache,
                                   previousCache: previousCache,
                                   activationOutputError: activationOutputError,
                                   nextActivationError: nextActivationError,
                                   nextCellError: ect,
                                   batchSize: 1,
                                   parameters: .init(forgetGateWeights: self.forgetGateWeights.detached(),
                                                     inputGateWeights: self.inputGateWeights.detached(),
                                                     gateGateWeights: self.gateGateWeights.detached(),
                                                     outputGateWeights: self.outputGateWeights.detached(),
                                                     forgetGateBiases: self.forgetGateBiases.detached(),
                                                     inputGateBiases: self.inputGateBiases.detached(),
                                                     gateGateBiases: self.gateGateBiases.detached(),
                                                     outputGateBiases: self.outputGateBiases.detached()))
      
      if wrtLSTMCellInputWeightsDerivatives.isEmpty {
        wrtLSTMCellInputWeightsDerivatives = backward.weights
      } else {
        wrtLSTMCellInputWeightsDerivatives = wrtLSTMCellInputWeightsDerivatives + backward.weights
      }
      
      if wrtLSTMCellInputBiasDerivatives.isEmpty {
        wrtLSTMCellInputBiasDerivatives = backward.biases
      } else {
        wrtLSTMCellInputBiasDerivatives = wrtLSTMCellInputBiasDerivatives + backward.biases
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

    // merge all biases into a giant 5 depth tensor, shape will be broken here
    let biasDerivatives = wrtLSTMCellInputBiasDerivatives.concat().concat(wrtOutputBiasesDerivatives, axis: 2)
    
    return (wrtEmbeddings, weightDerivatives, biasDerivatives)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = TensorSize(rows: 1,
                            columns: vocabSize,
                            depth: returnSequence ? batchLength : 1)
  }

  private func reset() {
    cellCache.clear()
  }

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
    
    let outputWeights = initializer.calculate(size: TensorSize(array: [hiddenUnits, vocabSize, 1]),
                                              input: inputUnits * vocabSize,
                                              out: inputUnits * vocabSize)
    
    
    self.outputGateWeights = outputGateWeights
    self.forgetGateWeights = forgetWeights
    self.gateGateWeights = gateWeights
    self.inputGateWeights = inputWeights
    self.hiddenOutputWeights = outputWeights
  }
  
  private func initializeBiases() {
    let biases = Tensor(NumSwift.zerosLike((rows: 1, columns: 1, depth: 1)))
    self.outputGateBiases = biases.detached()
    self.forgetGateBiases = biases.detached()
    self.gateGateBiases = biases.detached()
    self.inputGateBiases = biases.detached()
    
    self.hiddenOutputBiases = Tensor(NumSwift.zerosLike((rows: 1, columns: 1, depth: 1)))
  }
  
  private func setupInitialState() -> Cache {
    let a = Tensor(NumSwift.zerosLike((rows: 1, columns: hiddenUnits, depth: batchLength)))
    let c = Tensor(NumSwift.zerosLike((rows: 1, columns: hiddenUnits, depth: batchLength)))
    
    let og =  Tensor(NumSwift.zerosLike((rows: 1, columns: hiddenUnits, depth: 1)))
    let ig =  Tensor(NumSwift.zerosLike((rows: 1, columns: hiddenUnits, depth: 1)))
    let fg =  Tensor(NumSwift.zerosLike((rows: 1, columns: hiddenUnits, depth: 1)))
    let gg =  Tensor(NumSwift.zerosLike((rows: 1, columns: hiddenUnits, depth: 1)))

    let embedding =  Tensor(NumSwift.zerosLike((rows: 1, columns: inputUnits, depth: 1)))
    let output =  Tensor(NumSwift.zerosLike((rows: 1, columns: vocabSize, depth: 1)))

    let initialCache = Cache(lstm: .init(forgetGate: fg,
                                         inputGate: ig,
                                         outputGate: og,
                                         gateGate: gg),
                             cell: c,
                             activation: a,
                             embedding: embedding,
                             outputValue: output)

    return initialCache
  }

}

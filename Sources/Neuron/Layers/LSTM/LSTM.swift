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
  /// Weights tensor for the forget gate of the LSTM cell.
  public var forgetGateWeights: Tensor = Tensor()
  /// Biases tensor for the forget gate of the LSTM cell.
  public var forgetGateBiases: Tensor = Tensor()
  
  /// Weights tensor for the input gate of the LSTM cell.
  public var inputGateWeights: Tensor = Tensor()
  /// Biases tensor for the input gate of the LSTM cell.
  public var inputGateBiases: Tensor = Tensor()
  
  /// Weights tensor for the gate gate (cell gate) of the LSTM cell.
  public var gateGateWeights: Tensor = Tensor()
  /// Biases tensor for the gate gate (cell gate) of the LSTM cell.
  public var gateGateBiases: Tensor = Tensor()
  
  /// Weights tensor for the output gate of the LSTM cell.
  public var outputGateWeights: Tensor = Tensor()
  /// Biases tensor for the output gate of the LSTM cell.
  public var outputGateBiases: Tensor = Tensor()
  
  /// Weights tensor for projecting the hidden state to the output of the LSTM layer.
  public var hiddenOutputWeights: Tensor = Tensor()
  /// Biases tensor for projecting the hidden state to the output of the LSTM layer.
  public var hiddenOutputBiases: Tensor = Tensor()
  
  private var hiddenUnits: Int
  private var vocabSize: Int
  private var inputUnits: Int
  private var batchLength: Int
  private let returnSequence: Bool
  
  /// A concatenated view of all gate weight tensors (forget, input, gate, output, and hidden-output).
  ///
  /// Setting this property directly is not supported; use the individual gate weight properties instead.
  public override var weights: Tensor {
    get {
      forgetGateWeights.concat(inputGateWeights, axis: 2)
        .concat(gateGateWeights, axis: 2)
        .concat(outputGateWeights, axis: 2)
        .concat(hiddenOutputWeights, axis: 2)
    }
    set {
      fatalError("Please use the `gate` property instead to manage weights on LSTM layers")
    }
  }
  
  /// A concatenated view of all gate bias tensors (forget, input, gate, output, and hidden-output).
  ///
  /// Setting this property directly is not supported; use the individual gate bias properties instead.
  public override var biases: Tensor {
    get {
      forgetGateBiases.concat(inputGateBiases, axis: 0)
        .concat(gateGateBiases, axis: 0)
        .concat(outputGateBiases, axis: 0)
        .concat(hiddenOutputBiases, axis: 0)
    }
    set {
      fatalError("Please use the `gate` property instead to manage weights on LSTM layers")
    }
  }
  
  /// A container holding the activation tensors produced by each gate of an LSTM cell for a single time step.
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
  
  /// A cache storing intermediate values computed during an LSTM forward pass, used for backpropagation.
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
              vocabSize: Int,
              linkId: String = UUID().uuidString) {
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
               linkId: linkId,
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
  
  /// Encodes LSTM gate/output parameters and topology metadata.
  ///
  /// - Parameter encoder: Encoder used for serialization.
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
    var localCellCache: [Cache] = []
    
    let tensorContext = TensorContext { inputs, gradient, wrt in
      self.backward(inputs: inputs, gradient: gradient, cellCache: localCellCache)
    }
    
    var out = Tensor(context: tensorContext)
    
    let range = 0..<batchLength
    
    /// What happens to the prediction after we extend pass the batchLength?
    /// we need to truncate the input data if this happens to fit the expected window length
    for index in range {
      
      // get embeddings from input - use depth slice instead of .value
      let getEmbeddings: Tensor = index < tensor.size.depth
        ? tensor.depthSliceTensor(index)
      : Tensor.fillWith(value: 0, size: .init(rows: 1, columns: inputUnits, depth: 1))
      
      let cell = LSTMCell(hidden: hiddenUnits,
                          input: inputUnits,
                          vocabSize: vocabSize,
                          biasEnabled: biasEnabled)
      
      let cellParameters = LSTMCell.Parameters(forgetGateWeights: forgetGateWeights.detached(),
                                               inputGateWeights: inputGateWeights.detached(),
                                               gateGateWeights: gateGateWeights.detached(),
                                               outputGateWeights: outputGateWeights.detached(),
                                               forgetGateBiases: forgetGateBiases.detached(),
                                               inputGateBiases: inputGateBiases.detached(),
                                               gateGateBiases: gateGateBiases.detached(),
                                               outputGateBiases: outputGateBiases.detached())
      
      let previousCache = localCellCache[safe: index - 1, setupInitialState()]
      
      let cellOutput = cell.forward(tensor: getEmbeddings,
                                    context: context,
                                    parameters: cellParameters,
                                    previousCache: previousCache) // needs to be previous cache
      
      // used mainly for prediction and shouldn't be used in back propogation unless there's a gradient associated with it
      let outputCellParameters = OutputCell.Parameters(hiddenOutputWeights: hiddenOutputWeights.detached(),
                                                       hiddenOutputBiases: hiddenOutputBiases.detached(),
                                                       activationMatrix: cellOutput.activationMatrix.detached(),
                                                       vocabSize: vocabSize,
                                                       hiddenSize: hiddenUnits)
      
      let outputCell = OutputCell(device: device, biasEnabled: biasEnabled, parameters: outputCellParameters)
      
      
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
    
    if returnSequence == false, out.size.depth > 0 {
      let lastSlice = out.depthSlice(out.size.depth - 1)
      let lastSize = TensorSize(rows: out.size.rows, columns: out.size.columns, depth: 1)
      out = Tensor(lastSlice, size: lastSize, context: tensorContext)
    }
    
    out.setGraph(tensor)
    
    return super.forward(tensor: out, context: context)
  }
  
  
  /// Applies accumulated LSTM gate/output gradients to trainable parameters.
  ///
  /// - Parameters:
  ///   - gradients: Combined weight and bias gradient tensors.
  ///   - learningRate: Learning rate already reflected by optimizer gradient scaling.
  public override func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    /*
     order of weights in tensor...
     
     dForgetGateWeights = 0
     dInputGateWeights = 1
     dGateGateWeights = 2
     dOutputGateWeights = 3
     
     hiddenOutputWeightGradients = 4
     */
    
    // Split weight gradients along depth (each gate's weights are one depth slice)
    let gLayerTensors = gradients.weights
    
    if gLayerTensors.size.depth >= 5 {
      let forgetGateWeightGrads = gLayerTensors.depthSliceTensor(0)
      let inputGateWeightGrads = gLayerTensors.depthSliceTensor(1)
      let gateGateWeightGrads = gLayerTensors.depthSliceTensor(2)
      let outputGateWeightGrads = gLayerTensors.depthSliceTensor(3)
      
      let hiddenOutputWeightGradients = gLayerTensors[..<hiddenUnits, ..<vocabSize, 4...]
      
      forgetGateWeightGrads.l2Normalize()
      inputGateWeightGrads.l2Normalize()
      gateGateWeightGrads.l2Normalize()
      outputGateWeightGrads.l2Normalize()
      hiddenOutputWeightGradients.l2Normalize()
      
      self.forgetGateWeights = self.forgetGateWeights.copy() - forgetGateWeightGrads
      self.inputGateWeights = self.inputGateWeights.copy() - inputGateWeightGrads
      self.gateGateWeights = self.gateGateWeights.copy() - gateGateWeightGrads
      self.outputGateWeights = self.outputGateWeights.copy() - outputGateWeightGrads
      
      self.hiddenOutputWeights = self.hiddenOutputWeights.copy() - hiddenOutputWeightGradients
    }
    
    /*
     order of biases in tensor...
     
     dForgetGateBiases = 0
     dInputGateBiases = 1
     dGateGateBiases = 2
     dOutputGateBiases = 3
     
     hiddenOutputWeightBiases = 4
     */
    let gBiasLayerTensors = gradients.biases
    
    if biasEnabled,
       gBiasLayerTensors.size.depth >= 5 {
      
      let forgetGateBiasGrads = gBiasLayerTensors.depthSliceTensor(0)
      let inputGateBiasGrads = gBiasLayerTensors.depthSliceTensor(1)
      let gateGateBiasGrads = gBiasLayerTensors.depthSliceTensor(2)
      let outputGateBiasGrads = gBiasLayerTensors.depthSliceTensor(3)
      
      let hiddenOutputBiasGradients = gBiasLayerTensors[..<vocabSize, 0..., 4...]
      
      forgetGateBiases = forgetGateBiases.copy() - forgetGateBiasGrads
      inputGateBiases = inputGateBiases.copy() - inputGateBiasGrads
      gateGateBiases = gateGateBiases.copy() - gateGateBiasGrads
      outputGateBiases = outputGateBiases.copy() - outputGateBiasGrads
      hiddenOutputBiases = hiddenOutputBiases.copy() - hiddenOutputBiasGradients
    }
  }
  
  // MARK: Private
  private func backward(inputs: Tensor, gradient: Tensor, cellCache: [Cache]) -> TensorContext.TensorBackpropResult {
    // eat and ect are kept as [[Scalar]] for LSTMCell.backward interface compatibility
    var eat: Tensor = .fillWith(value: 0, size: .init(rows: 1, columns: hiddenUnits, depth: 1))
    var ect: Tensor = eat.copy()
    
    var wrtOutputWeightsDerivatives: Tensor = Tensor()
    var wrtOutputBiasesDerivatives: Tensor = Tensor()
    var wrtLSTMCellInputWeightsDerivatives: LSTMCell.ParameterDerivatives = .init()
    var wrtLSTMCellInputBiasDerivatives: LSTMCell.ParameterDerivatives = .init()
    
    var wrtEmbeddingsTensor = Tensor()
    
    for index in (0..<cellCache.count).reversed() {
      
      let cache = cellCache[index]
      let previousCache = cellCache[safe: index - 1]
      
      // Get delta for this timestep from gradient depth slices
      let delta: Tensor = index < gradient.size.depth
        ? gradient.depthSliceTensor(index)
        : gradient.zerosLike().depthSliceTensor(0)
      
      let activationErrors = cache.outputValue.gradients(delta: delta,
                                                         wrt: cache.activation)
      
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

      let activationInputTensor = activationErrors.input[safe: 0, Tensor()]
      
      let activationOutputErrorTensor = activationInputTensor.depthSliceTensor(0)

      let cell = LSTMCell(hidden: self.hiddenUnits,
                          input: self.inputUnits,
                          vocabSize: self.vocabSize,
                          biasEnabled: biasEnabled,
                          device: self.device)
      
      let backward = cell.backward(cache: cache,
                                   previousCache: previousCache,
                                   activationOutputError: activationOutputErrorTensor,
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
      
      // Accumulate embeddings using Tensor concat (prepend for reversed order)
      if wrtEmbeddingsTensor.isEmpty {
        wrtEmbeddingsTensor = embeddingError
      } else {
        wrtEmbeddingsTensor = embeddingError.concat(wrtEmbeddingsTensor, axis: 2)
      }
      
      // Convert Tensor depth slices to [[Scalar]] for LSTMCell interface
      if previousActivationError.size.depth > 0 && previousCellError.size.depth > 0 {
        let paeSlice = previousActivationError.depthSliceTensor(0)
        let pceSlice = previousCellError.depthSliceTensor(0)
        eat = paeSlice
        ect = pceSlice
      }
    }
        
    // merge all weights into a giant 5 depth tensor, shape will be broken here
    let weightDerivatives = wrtLSTMCellInputWeightsDerivatives.concat().concat(wrtOutputWeightsDerivatives, axis: 2)
    
    // merge all biases into a giant 5 depth tensor, shape will be broken here
    let biasDerivatives = wrtLSTMCellInputBiasDerivatives.concat().concat(wrtOutputBiasesDerivatives, axis: 2)
    
    // Normalize gradients by sequence length to prevent explosion
    // This is standard practice for RNNs - gradients are accumulated across timesteps,
    // so we normalize by the number of timesteps to get average gradients
    var normalizedWeightDerivatives = weightDerivatives
    var normalizedBiasDerivatives = biasDerivatives
    var normalizedEmbeddings = wrtEmbeddingsTensor
    
    let sequenceLength = Tensor.Scalar(cellCache.count)

    if sequenceLength > 1 {
      normalizedBiasDerivatives = normalizedBiasDerivatives.copy() / sequenceLength
      normalizedWeightDerivatives = normalizedWeightDerivatives.copy() / sequenceLength
      normalizedEmbeddings = normalizedEmbeddings.copy() / sequenceLength
    }
    
    return (normalizedEmbeddings, normalizedWeightDerivatives, normalizedBiasDerivatives)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = TensorSize(rows: 1,
                            columns: vocabSize,
                            depth: returnSequence ? batchLength : 1)
  }
  
  private func initializeWeights() {
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
    let biases = Tensor(NumSwift.zerosLike((rows: 1, columns: hiddenUnits, depth: 1)))
    self.outputGateBiases = biases.copy()
    self.gateGateBiases = biases.copy()
    self.inputGateBiases = biases.copy()
    // Initialize forget gate bias to 1.0 to help gradient flow
    // This encourages the LSTM to remember information by default
    self.forgetGateBiases = Tensor(NumSwift.onesLike((rows: 1, columns: hiddenUnits, depth: 1)))
    
    self.hiddenOutputBiases = Tensor(NumSwift.zerosLike((rows: 1, columns: vocabSize, depth: 1)))
  }
  
  private func setupInitialState() -> Cache {
    let zeroTensor = Tensor(NumSwift.zerosLike((rows: 1, columns: hiddenUnits, depth: 1)))
    
    let a = zeroTensor.copy()
    let c = zeroTensor.copy()
    
    let og =  zeroTensor.copy()
    let ig =  zeroTensor.copy()
    let fg =  zeroTensor.copy()
    let gg =  zeroTensor.copy()
    
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

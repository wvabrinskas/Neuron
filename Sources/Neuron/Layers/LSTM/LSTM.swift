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
  
  /// Maximum L2 norm allowed for the hidden-state and cell-state errors carried backward
  /// between timesteps during BPTT. Each timestep multiplies these errors by the recurrent
  /// weights and gate derivatives, so without a bound they grow geometrically with sequence
  /// length. When either error's norm exceeds this value it is rescaled (direction preserved)
  /// before being handed to the previous timestep. `nil` (the default) disables the bound.
  ///
  /// With correct gradients the LSTM trains stably without it; it is an opt-in safety net for
  /// very long sequences. It bounds the gradient at its source, which a global
  /// `Optimizer.gradientClip` cannot do in front of a scale-invariant optimizer such as `Adam`.
  public var recurrentErrorClip: Tensor.Scalar?
  
  /// A flat, concatenated view of all gate weight tensors (forget, input, gate, output, and hidden-output).
  ///
  /// The groups have different shapes (gates are `(inputUnits + hiddenUnits) x hiddenUnits`, the
  /// hidden-output projection is `vocabSize x hiddenUnits`), so they are packed along axis `-1`
  /// into a 1-D tensor whose declared size always equals its storage. Slice with
  /// `splitWeightGradients` rather than by depth.
  ///
  /// Setting this property directly is not supported; use the individual gate weight properties instead.
  public override var weights: Tensor {
    get {
      forgetGateWeights.concat(inputGateWeights, axis: -1)
        .concat(gateGateWeights, axis: -1)
        .concat(outputGateWeights, axis: -1)
        .concat(hiddenOutputWeights, axis: -1)
    }
    set {
      fatalError("Please use the `gate` property instead to manage weights on LSTM layers")
    }
  }
  
  /// A flat, concatenated view of all gate bias tensors (forget, input, gate, output, and hidden-output),
  /// packed along axis `-1` for the same reason as `weights`.
  ///
  /// Setting this property directly is not supported; use the individual gate bias properties instead.
  public override var biases: Tensor {
    get {
      forgetGateBiases.concat(inputGateBiases, axis: -1)
        .concat(gateGateBiases, axis: -1)
        .concat(outputGateBiases, axis: -1)
        .concat(hiddenOutputBiases, axis: -1)
    }
    set {
      fatalError("Please use the `gate` property instead to manage weights on LSTM layers")
    }
  }
  
  /// A container holding the activation tensors produced by each gate of an LSTM cell for a single time step.
  public class LSTMActivations {
    /// Forget-gate activation tensor for this time step.
    let forgetGate: Tensor
    /// Input-gate activation tensor for this time step.
    let inputGate: Tensor
    /// Output-gate activation tensor for this time step.
    let outputGate: Tensor
    /// Gate-gate (cell-input) activation tensor for this time step.
    let gateGate: Tensor

    /// Creates activations from a raw `LSTMCell.Activations` value.
    /// - Parameter activations: The cell activations from which to copy gate tensors.
    init(activations: LSTMCell.Activations) {
      self.forgetGate = activations.fa
      self.inputGate = activations.ia
      self.outputGate = activations.oa
      self.gateGate = activations.ga
    }
    
    /// Creates default-empty gate activations.
    /// - Parameters:
    ///   - forgetGate: Forget-gate activation. Defaults to an empty tensor.
    ///   - inputGate: Input-gate activation. Defaults to an empty tensor.
    ///   - outputGate: Output-gate activation. Defaults to an empty tensor.
    ///   - gateGate: Gate-gate activation. Defaults to an empty tensor.
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
    /// Gate activations recorded during the forward pass for this time step.
    var lstm: LSTMActivations
    /// Cell memory state tensor at this time step.
    var cell: Tensor
    /// Hidden-state activation tensor at this time step.
    var activation: Tensor
    /// Embedded input vector at this time step.
    var embedding: Tensor
    /// Output cell used to project the hidden state to vocabulary logits.
    var output: OutputCell?
    /// Projected output (softmax probabilities) at this time step.
    var outputValue: Tensor

    /// Creates a cache entry for a single LSTM time step.
    /// - Parameters:
    ///   - lstm: Gate activations for this time step.
    ///   - cell: Cell memory state tensor.
    ///   - activation: Hidden-state activation tensor.
    ///   - embedding: Embedded input for this time step.
    ///   - output: Optional output cell holding projection weights.
    ///   - outputValue: Projected vocabulary probability tensor.
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
    
    /// Updates cache fields with new values, preserving existing values for any `nil` arguments.
    /// - Parameters:
    ///   - lstm: Replacement gate activations, or `nil` to keep current.
    ///   - cell: Replacement cell state tensor, or `nil` to keep current.
    ///   - activation: Replacement hidden-state tensor, or `nil` to keep current.
    ///   - embedding: Replacement embedded input, or `nil` to keep current.
    ///   - output: Replacement output cell, or `nil` to keep current.
    ///   - outputValue: Replacement projected output, or `nil` to keep current.
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
  ///   - biasEnabled: Whether the gate and output biases are used. Default: `false`
  ///   - initializer: Initializer funciton to use
  ///   - hiddenUnits: Number of hidden use
  ///   - vocabSize: size of the expected vocabulary
  ///   - recurrentErrorClip: Optional maximum L2 norm of the hidden/cell errors carried between timesteps during BPTT. Default: `nil` (disabled)
  ///   - linkId: Unique identifier used to link this layer's weights across serialization. Defaults to a new UUID string.
  public init(inputUnits: Int,
              batchLength: Int,
              returnSequence: Bool = true,
              biasEnabled: Bool = false,
              initializer: InitializerType = .xavierNormal,
              hiddenUnits: Int,
              vocabSize: Int,
              recurrentErrorClip: Tensor.Scalar? = nil,
              linkId: String = UUID().uuidString) {
    let inputSize = TensorSize(rows: 1,
                               columns: vocabSize,
                               depth: batchLength)
    self.hiddenUnits = hiddenUnits
    self.vocabSize = vocabSize
    self.inputUnits = inputUnits
    self.batchLength = batchLength
    self.returnSequence = returnSequence
    self.recurrentErrorClip = recurrentErrorClip
    
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
         inputUnits,
         recurrentErrorClip,
         linkId
  }
  
  /// Decodes an LSTM layer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  convenience required public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    let hiddenUnits = try container.decodeIfPresent(Int.self, forKey: .hiddenUnits) ?? 0
    let vocabSize = try container.decodeIfPresent(Int.self, forKey: .vocabSize) ?? 0
    let inputUnits = try container.decodeIfPresent(Int.self, forKey: .inputUnits) ?? 0
    let batchLength = try container.decodeIfPresent(Int.self, forKey: .batchLength) ?? 0
    
    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    
    self.init(inputUnits: inputUnits,
              batchLength: batchLength,
              hiddenUnits: hiddenUnits,
              vocabSize: vocabSize,
              linkId: linkId)
    
    self.biasEnabled = try container.decodeIfPresent(Bool.self, forKey: .biasEnabled) ?? false
    // Absent key (older exports) or explicit `null` both mean disabled.
    if container.contains(.recurrentErrorClip) {
      self.recurrentErrorClip = try container.decodeIfPresent(Tensor.Scalar.self, forKey: .recurrentErrorClip)
    }
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
  /// - Throws: An error if any values fail to encode.
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
    try container.encode(recurrentErrorClip, forKey: .recurrentErrorClip)
    try container.encode(linkId, forKey: .linkId)
  }
  
  
  /// The forward path for the LSTM layer. Should be preceeded by an `Embedding` layer.
  /// Emdedding input size expected is `(rows: 1, columns: inputUnits, depth: batchLength)`
  /// - Parameters:
  ///   - tensor: The `Embedding` input `Tensor`
  ///   - context: Network context carrying batch metadata through the forward pass.
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
     order of groups in the flat gradient tensor...
     
     dForgetGateWeights, dInputGateWeights, dGateGateWeights, dOutputGateWeights, hiddenOutputWeightGradients
     */
    if let grads = splitWeightGradients(gradients.weights) {
      // NOTE: these gradients arrive already scaled by the optimizer (Adam bakes the
      // learning rate into the delta), exactly like `Dense.apply`. Renormalizing them
      // here would discard that scaling and turn every update into a fixed unit-L2-norm
      // step, which never anneals as the gradient shrinks. Gradient growth across the
      // sequence is bounded at its source by `recurrentErrorClip` in `backward`.
      self.forgetGateWeights = self.forgetGateWeights.copy() - grads.forget
      self.inputGateWeights = self.inputGateWeights.copy() - grads.input
      self.gateGateWeights = self.gateGateWeights.copy() - grads.gate
      self.outputGateWeights = self.outputGateWeights.copy() - grads.output
      
      self.hiddenOutputWeights = self.hiddenOutputWeights.copy() - grads.hiddenOutput
    }
    
    // biases use the same group order as the weights
    if biasEnabled, let grads = splitBiasGradients(gradients.biases) {
      forgetGateBiases = forgetGateBiases.copy() - grads.forget
      inputGateBiases = inputGateBiases.copy() - grads.input
      gateGateBiases = gateGateBiases.copy() - grads.gate
      outputGateBiases = outputGateBiases.copy() - grads.output
      hiddenOutputBiases = hiddenOutputBiases.copy() - grads.hiddenOutput
    }
  }
  
  // MARK: Private
  
  /// One tensor per parameter group, in the order the LSTM packs its gradients.
  private struct GateGradients {
    let forget: Tensor
    let input: Tensor
    let gate: Tensor
    let output: Tensor
    let hiddenOutput: Tensor
  }
  
  /// Unpacks a flat gradient tensor into one tensor per group, each shaped like the matching
  /// parameter in `shapes`. Returns `nil` unless the storage is exactly the packed length.
  private func splitPacked(_ gradients: Tensor, shapes: [Tensor]) -> GateGradients? {
    let counts = shapes.map { $0.storage.count }
    guard counts.allSatisfy({ $0 > 0 }),
          gradients.storage.count == counts.reduce(0, +) else { return nil }
    
    var offset = 0
    var groups: [Tensor] = []
    groups.reserveCapacity(shapes.count)
    
    for shape in shapes {
      let count = shape.storage.count
      let slice = TensorStorage.create(count: count)
      slice.pointer.update(from: gradients.storage.pointer + offset, count: count)
      groups.append(Tensor(storage: slice, size: shape.size))
      offset += count
    }
    
    return GateGradients(forget: groups[0],
                         input: groups[1],
                         gate: groups[2],
                         output: groups[3],
                         hiddenOutput: groups[4])
  }
  
  /// Splits a flat weight-gradient tensor packed in `weights` order.
  private func splitWeightGradients(_ gradients: Tensor) -> GateGradients? {
    splitPacked(gradients, shapes: [forgetGateWeights,
                                    inputGateWeights,
                                    gateGateWeights,
                                    outputGateWeights,
                                    hiddenOutputWeights])
  }
  
  /// Splits a flat bias-gradient tensor packed in `biases` order.
  private func splitBiasGradients(_ gradients: Tensor) -> GateGradients? {
    splitPacked(gradients, shapes: [forgetGateBiases,
                                    inputGateBiases,
                                    gateGateBiases,
                                    outputGateBiases,
                                    hiddenOutputBiases])
  }
  
  /// Rescales `error` so its L2 norm does not exceed `recurrentErrorClip`. Direction is preserved.
  private func clipRecurrentError(_ error: Tensor) -> Tensor {
    guard let clip = recurrentErrorClip else { return error }
    let norm = error.l2Norm()
    guard norm > clip else { return error }
    // Storage-level arithmetic: the carried error needs no autograd graph.
    return Tensor(storage: error.storage * (clip / norm), size: error.size)
  }
  
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
      
      // Bound the errors carried into the previous timestep so BPTT cannot grow geometrically.
      if previousActivationError.size.depth > 0 && previousCellError.size.depth > 0 {
        eat = clipRecurrentError(previousActivationError.depthSliceTensor(0))
        ect = clipRecurrentError(previousCellError.depthSliceTensor(0))
      }
    }
        
    // Pack every group into one flat tensor (axis -1) in the same order as `weights` / `biases`.
    // The hidden-output group has a different shape from the gates, so a depth concat would
    // produce a tensor whose declared size exceeds its storage; flat packing keeps them equal.
    let weightDerivatives = wrtLSTMCellInputWeightsDerivatives.concat().concat(wrtOutputWeightsDerivatives, axis: -1)
    let biasDerivatives = wrtLSTMCellInputBiasDerivatives.concat().concat(wrtOutputBiasesDerivatives, axis: -1)
    
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
  
  /// Called by `Sequential.compile()` when input size is propagated; sets the LSTM output shape based on vocab and sequence settings.
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

// MARK: - GradientNormInspectable

extension LSTM: GradientNormInspectable {
  /// Reports one L2 norm per gate plus the hidden-output projection, using the same flat
  /// packed layout `apply(gradients:learningRate:)` reads.
  public func gradientNormBreakdown(weights: Tensor, biases: Tensor) -> [GradientNormReport.Group] {
    let w = splitWeightGradients(weights)
    let b = biasEnabled ? splitBiasGradients(biases) : nil
    
    func norm(_ tensor: Tensor?) -> Tensor.Scalar {
      guard let tensor, tensor.isEmpty == false else { return 0 }
      return tensor.l2Norm()
    }
    
    return [("forgetGate", norm(w?.forget), norm(b?.forget)),
            ("inputGate", norm(w?.input), norm(b?.input)),
            ("gateGate", norm(w?.gate), norm(b?.gate)),
            ("outputGate", norm(w?.output), norm(b?.output)),
            ("hiddenOutput", norm(w?.hiddenOutput), norm(b?.hiddenOutput))]
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/26/22.
//

import Foundation
import NumSwift
import NumSwiftC
import Atomics

/// Layer types
public enum EncodingType: String, Codable {
  case leakyRelu,
       relu,
       sigmoid,
       softmax,
       swish,
       tanh,
       batchNormalize,
       conv2d,
       dense,
       dropout,
       flatten,
       maxPool,
       reshape,
       transConv2d,
       layerNormalize,
       lstm,
       embedding,
       avgPool,
       selu,
       resNet,
       globalAvgPool,
       depthwiseConv2d,
       instanceNorm,
       rexNet,
       add,
       multiply,
       none
}

/// A layer that performs an activation function
public protocol ActivationLayer: Layer {
  var type: Activation { get }
}

/// A layer that performs a convolution operation
public protocol ConvolutionalLayer: Layer {
  var filterCount: Int { get }
  var filters: [Tensor] { get }
  var filterSize: (rows: Int, columns: Int) { get }
  var strides: (rows: Int, columns: Int) { get }
  var padding: NumSwift.ConvPadding { get }
}

/// A batch of tensors used as input or output for a layer.
public typealias TensorBatch = [Tensor]
/// The the object that perform ML operations
public protocol Layer: AnyObject, Codable {
  var details: String { get }
  var encodingType: EncodingType { get set }
  var extraEncodables: [String: Codable]? { get }
  var inputSize: TensorSize { get set }
  var outputSize: TensorSize { get }
  var weights: Tensor { get }
  var biases: Tensor { get }
  var biasEnabled: Bool { get set }
  var trainable: Bool { get set }
  var isTraining: Bool { get set }
  var initializer: Initializer { get }
  var device: Device { get set }
  var usesOptimizer: Bool { get set }
  var batchSize: Int { get set }
  var linkId: String { get }
  @discardableResult
  /// Runs the layer's forward transformation for a single tensor.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context for batch/thread metadata.
  /// - Returns: Output tensor for this layer.
  func forward(tensor: Tensor, context: NetworkContext) -> Tensor
  /// Runs the layer's forward transformation for a tensor batch.
  ///
  /// - Parameters:
  ///   - tensorBatch: Input batch.
  ///   - context: Network execution context for batch/thread metadata.
  /// - Returns: Output batch in input order.
  func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch
  /// Applies parameter updates to the layer.
  ///
  /// - Parameters:
  ///   - gradients: Gradient tuple for this layer.
  ///   - learningRate: Optimizer learning-rate scalar.
  func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar)
  /// Exports trainable parameter tensors for persistence.
  ///
  /// - Returns: Layer-owned parameter tensors.
  /// - Throws: `LayerErrors` when weights are not initialized.
  func exportWeights() throws -> [Tensor]
  /// Imports trainable parameter tensors for this layer.
  ///
  /// - Parameter weights: Parameter tensors matching the layer's expected shapes.
  /// - Throws: `LayerErrors` when tensor counts or shapes are invalid.
  func importWeights(_ weights: [Tensor]) throws
}

/// Errors that can occur during layer operations such as weight importing.
public enum LayerErrors: Error, LocalizedError {
  case weightImportError
  case generic(error: String)
  
/// A human-readable description of the error that occurred.
  public var errorDescription: String? {
    switch self {
    case .weightImportError:
      return "Unable to import weights. Please validate the shapes are correct"
    case .generic(let error):
      return error
    }
  }
}

extension Layer {
/// Default implementation of `extraEncodables` returning an empty dictionary.
  public var extraEncodables: [String: Codable]? {
    return [:]
  }
}

open class ArithmecticLayer: BaseLayer {
  // looks up through the tensor input graph to find the first input tensor with this label applied.
  // and applies the arithmetic to it that the layer defines along with the input to this layer
  var linkTo: String
  
  override public var usesOptimizer: Bool { get { false } set { } }

  init(inputSize: TensorSize? = nil,
       initializer: InitializerType = Constants.defaultInitializer,
       biasEnabled: Bool = false,
       encodingType: EncodingType,
       linkId: String = UUID().uuidString,
       linkTo: String) {
    self.linkTo = linkTo
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
               linkId: linkId,
               encodingType: encodingType)
  }

  enum CodingKeys: String, CodingKey {
    case inputSize, type, linkTo
  }
  
  open func function(input: Tensor, other: Tensor) -> Tensor {
    fatalError("override in subclass")
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    /// do something when the input size is set when calling `compile` on `Sequential`
    /// like setting the output size or initializing the weights
    outputSize = inputSize
  }
  
  required convenience public init(from decoder: Decoder) throws {
    self.init(encodingType: .add, linkTo: "")
  }
  
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkTo, forKey: .linkTo)
  }

  func lookupInput(input: Tensor) -> Tensor? {
    if input.label == linkTo { return input }

    let out = input.graph.first(where: { $0.value.label.contains(linkTo) })
    return out?.value
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    
    guard let other = lookupInput(input: tensor) else {
      assertionFailure("could not find reference input tensor in graph")
      return Tensor()
    }
    
    let out = function(input: tensor, other: other)
    
    return super.forward(tensor: out, context: context)
  }
  
}

open class BaseLayer: Layer {
/// A base class providing default implementations of common `Layer` properties and behaviors.
  public var details: String {
    """
    Input: \(formatTensorSize(inputSize)) → Output: \(formatTensorSize(outputSize))
    """
  }
  
/// A human-readable summary of the layer's input and output sizes.
  public var encodingType: EncodingType
/// The encoding type used to identify this layer during serialization.
  public var inputSize: TensorSize = .init() {
    didSet {
      onInputSizeSet()
    }
  }
/// The input tensor size for this layer. Setting this triggers `onInputSizeSet()` to reconfigure the layer.
  public var outputSize: TensorSize = .init()
/// The output tensor size produced by this layer.
  public var weights: Tensor = .init()
/// The learnable weight parameters of this layer.
  public var biases: Tensor = .init()
/// The learnable bias parameters of this layer.
  public var biasEnabled: Bool = false
/// Whether bias parameters are applied during the forward pass.
  public var trainable: Bool = true
/// Whether this layer's parameters are updated during training.
  public var isTraining: Bool = true
/// Whether this layer is currently in training mode, affecting behaviors such as dropout.
  public var initializer: Initializer
/// The weight initializer strategy used to initialize this layer's parameters.
  public var device: Device = CPU()
/// The compute device (e.g., CPU or GPU) used to execute this layer's operations.
  public var batchSize: Int = 1 {
    didSet {
      onBatchSizeSet()
    }
  }
  
  /// Set this to reference the output of this layer in an arithmetic layer. eg a Shortcut path
  public var linkId: String = UUID().uuidString
  
  // defines whether the gradients are run through the optimizer before being applied.
  // this could be useful if a layer manages its own weight updates
/// The number of samples processed in a single forward/backward pass. Setting this triggers `onBatchSizeSet()`.
  public var usesOptimizer: Bool = true
  
  /// Creates a new base layer configuration.
  ///
  /// - Parameters:
  ///   - inputSize: Optional known input shape for eager setup.
  ///   - initializer: Weight initializer strategy.
  ///   - biasEnabled: Whether the layer should use bias parameters.
  ///   - encodingType: Serialized layer type identifier.
  ///   - linkId: Set this to reference the output of this layer in an arithmetic layer. eg a Shortcut path
  public init(inputSize: TensorSize? = nil,
              initializer: InitializerType = Constants.defaultInitializer,
              biasEnabled: Bool = false,
              linkId: String = UUID().uuidString,
              encodingType: EncodingType) {
    self.inputSize = inputSize ?? TensorSize(array: [])
    self.initializer = initializer.build()
    self.biasEnabled = biasEnabled
    self.encodingType = encodingType
    self.linkId = linkId
    
    if inputSize != nil {
      onInputSizeSet()
    }
  }
  
  @discardableResult
  /// Convenience call-syntax wrapper around `forward(tensor:context:)`.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Layer output tensor.
  open func callAsFunction(_ tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    forward(tensor: tensor, context: context)
  }
  
  required convenience public init(from decoder: Decoder) throws {
    // override
    self.init(encodingType: .none)
  }
  
  /// Encodes layer configuration for persistence.
  ///
  /// Subclasses should override and encode their own fields.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public func encode(to encoder: Encoder) throws {
    // override
  }
  
  /// Default batch forward implementation that iterates over each tensor.
  ///
  /// - Parameters:
  ///   - tensorBatch: Input batch.
  ///   - context: Network execution context.
  /// - Returns: Layer outputs for each input tensor.
  public func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    var result: TensorBatch = []
    
    for tensor in tensorBatch {
      result.append(forward(tensor: tensor, context: context))
    }
    
    return result
  }
  
  @discardableResult
  /// Default single-tensor forward placeholder.
  ///
  /// Subclasses must override with the layer's actual forward math.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Placeholder tensor.
  public func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    // override
    tensor.label = encodingType.rawValue + "-" + linkId
    return tensor
  }
  
  // guarenteed to be single threaded operation
  /// Default parameter application placeholder.
  ///
  /// Subclasses should override to update weights/biases from gradients.
  ///
  /// - Parameters:
  ///   - gradients: Gradient tuple for this layer.
  ///   - learningRate: Optimizer learning rate.
  public func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    // override
  }
  
  // MARK: Internal
  /// Lifecycle hook invoked when `inputSize` changes.
  public func onInputSizeSet() {
    // override
  }
  
  /// Lifecycle hook invoked when `batchSize` changes.
  public func onBatchSizeSet() {
    // override
  }
  
  /// Exports this layer's primary weight tensor.
  ///
  /// - Returns: Single weight tensor for simple layers.
  /// - Throws: `LayerErrors` if weights are not initialized.
  public func exportWeights() throws -> [Tensor] {
    guard self.weights.isEmpty == false else {
      throw LayerErrors.generic(error: "\(encodingType.rawValue.capitalized) weights have not been initialized.")
    }
    
    return [weights]
  }
  
  /// Imports this layer's primary weight tensor.
  ///
  /// - Parameter weights: Array containing exactly one weight tensor.
  /// - Throws: `LayerErrors` when count/shape are invalid.
  public func importWeights(_ weights: [Tensor]) throws {
    guard self.weights.isEmpty == false else {
      throw LayerErrors.generic(error: "\(encodingType.rawValue.capitalized) weights have not been initialized.")
    }
    
    guard weights.count == 1, let weight = weights[safe: 0] else {
      throw(LayerErrors.generic(error: "\(encodingType.rawValue.capitalized) expects only one Tensor in the array. Got: \(weights.count)"))
    }
    
    try validateWeight(weight, against: self.weights)

    self.weights = weight
  }

  func validateWeight(_ weight: Tensor, against: Tensor) throws {
    let incomingShape = weight.shape
    let currentShape = against.shape
    
    guard incomingShape == currentShape else {
      throw(LayerErrors.generic(error: "\(encodingType.rawValue.capitalized) expects weights of shape: \(currentShape). Got: \(incomingShape)"))
    }
  }
  
  func sumBranchGradients(_ gradient: Tensor, to: Tensor) -> Tensor {
    var result = gradient.copy()
    for g in to.branchGradients {
      result = result.copy() + g.value
    }
    
    return result
  }
  
  private func formatTensorSize(_ size: TensorSize) -> String {
    let array = size.asArray
    if array.count <= 1 {
      return "\(array.first ?? 0)"
    }
    return array.map(String.init).joined(separator: "×")
  }
  
}

open class BaseConvolutionalLayer: BaseLayer, ConvolutionalLayer {
/// Whether gradients are passed through the optimizer before being applied. Set to `false` if the layer manages its own weight updates.
  public override var details: String {
    super.details +
    """
    \n
    Filters: \(filterCount)
    Strides: \(strides.rows)x\(strides.columns)
    Padding: \(padding.asString)
    """
  }
  
/// A human-readable summary including filter count, strides, and padding configuration.
  public override var weights: Tensor {
    get {
      var reduce = filters
      let first = reduce.removeFirst()
      
      let out = reduce.reduce(first) { partialResult, new in
        partialResult.concat(new, axis: 2)
      }.storage
      
      return Tensor(out, size: .init(rows: filterSize.rows,
                                     columns: filterSize.columns,
                                     depth: filterCount * inputSize.depth))
      
    }
    set {
      fatalError("Please use the `filters` property instead to manage weights on Convolutional layers")
    }
  }
/// A combined tensor representation of all convolutional filters, concatenated along the depth axis.
  public var filterCount: Int
/// The number of convolutional filters applied at this layer.
  public var filters: [Tensor] = []
/// The collection of filter tensors used for convolution.
  public var filterSize: (rows: Int, columns: Int)
/// The spatial dimensions (rows and columns) of each filter kernel.
  public var strides: (rows: Int, columns: Int)
/// The step size (rows and columns) used when sliding the filter over the input.
  public var padding: NumSwift.ConvPadding
  
  /// Default initializer for a 2d convolutional layer
  /// - Parameters:
  ///   - filterCount: Number of filters at this layer
  ///   - inputSize: Optional input size at this layer. If this is the first layer you will need to set this.
  ///   - strides: Number of row and column strides when performing convolution. Default: `(3,3)`
  ///   - padding: Padding type when performing the convolution. Default: `.valid`
  ///   - filterSize: Size of the filter kernel. Default: `(3,3)`
  ///   - initializer: Weight / filter initializer function. Default: `.heNormal`
  ///   - biasEnabled: Boolean defining if the filters have a bias applied. Default: `false`
  public init(filterCount: Int,
              inputSize: TensorSize? = nil,
              strides: (rows: Int, columns: Int) = (1,1),
              padding: NumSwift.ConvPadding = .valid,
              filterSize: (rows: Int, columns: Int) = (3,3),
              initializer: InitializerType = .heNormal,
              biasEnabled: Bool = false,
              linkId: String = UUID().uuidString,
              encodingType: EncodingType) {
    
    self.filterCount = filterCount
    self.strides = strides
    self.padding = padding
    self.filterSize = filterSize
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
               linkId: linkId,
               encodingType: encodingType)
    
    if biasEnabled {
      biases = Tensor([Tensor.Scalar](repeating: 0, count: filterCount))
    }
  }
  
  required convenience public init(from decoder: Decoder) throws {
    // override
    self.init(filterCount: 0, encodingType: .none)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    initializeFilters()
  }
  
  /// Exports convolution filters as the trainable weight set.
  ///
  /// - Returns: Filter tensors in filter index order.
  /// - Throws: `LayerErrors` if filters are not initialized.
  public override func exportWeights() throws -> [Tensor] {
    guard filters.isEmpty == false else {
      throw LayerErrors.generic(error: "\(encodingType.rawValue.capitalized) weights have not been initialized.")
    }
    
    return filters
  }
  
  /// Imports convolution filters for this layer.
  ///
  /// - Parameter weights: Filter tensors matching existing filter shapes.
  /// - Throws: `LayerErrors` if filters are missing or shape mismatches occur.
  public override func importWeights(_ weights: [Tensor]) throws {
    guard filters.isEmpty == false else {
      throw LayerErrors.generic(error: "\(encodingType.rawValue.capitalized) weights have not been initialized.")
    }
    
    for i in 0..<filters.count {
      let currentFilter = filters[i]
      let incomingFilter = weights[i]
      try validateWeight(currentFilter, against: incomingFilter)
    }
    
    filters = weights
  }
  
  func initializeFilters() {
    guard filters.isEmpty else {
      return
    }
    
    for _ in 0..<filterCount {
      let filter = initializer.calculate(size: .init(rows: filterSize.rows,
                                                     columns: filterSize.columns,
                                                     depth: inputSize.depth),
                                         input: inputSize.depth * filterSize.rows * filterSize.columns,
                                         out: outputSize.depth * filterSize.rows * filterSize.columns)
      
      filters.append(filter)
    }
  }
}

open class BaseActivationLayer: BaseLayer, ActivationLayer {
  
/// The padding strategy applied to the input before convolution.
  public override var details: String {
      ""
  }
  
/// The activation function type applied by this layer.
  public let type: Activation

  /// Creates a base activation layer.
  ///
  /// - Parameters:
  ///   - inputSize: Optional known input shape.
  ///   - type: Activation function represented by this layer.
  ///   - encodingType: Serialized layer type identifier.
  public init(inputSize: TensorSize? = nil,
              type: Activation,
              linkId: String = UUID().uuidString,
              encodingType: EncodingType) {
    self.type = type
    super.init(inputSize: inputSize,
               biasEnabled: false,
               linkId: linkId,
               encodingType: encodingType)
    
    self.usesOptimizer = false
  }
  
  required convenience public init(from decoder: Decoder) throws {
    self.init(inputSize: .init(),
              type: .none,
              encodingType: .none)
  }
  
  /// Applies the activation function and builds backpropagation context.
  ///
  /// - Parameters:
  ///   - tensor: Input tensor.
  ///   - context: Network execution context.
  /// - Returns: Activated output tensor.
  @discardableResult
/// Performs the forward pass by applying the layer's activation function to the input tensor and building the backward graph.
  /// - Parameter tensor: The input tensor to activate.
  /// - Parameter context: The network context for the current pass. Defaults to a new context.
  /// - Returns: A new tensor containing the activated values with a configured backward context.
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    
    let context = TensorContext { inputs, gradient, wrt in
      let derivResult = self.device.derivate(inputs, self.type)
      let outTensor = derivResult * gradient
      outTensor.label = self.type.asString() + "_input_grad"
      return (outTensor, Tensor(), Tensor())
    }
    
    let result = device.activate(tensor, type)
    let out = Tensor(result.storage, size: result.size, context: context)

    out.setGraph(tensor)
    out.label = encodingType.rawValue
    return out
  }
  
  override public func importWeights(_ weights: [Tensor]) throws {
    // no op
  }
  
  override public func exportWeights() throws -> [Tensor] {
    [Tensor()]
  }
}

extension NumSwift.ConvPadding {
  var asString: String {
    switch self {
    case .valid: return "valid"
    case .same: return "same"
    }
  }
}



open class BaseThreadBatchingLayer: BaseLayer {
  let updateLock = NSLock()
  let iterations = ManagedAtomic<Int>(0)

  override public var isTraining: Bool {
    didSet {
      if isTraining == false {
        iterations.store(0, ordering: .relaxed)
      }
    }
  }

  open var shouldPerformBatching: Bool {
    isTraining
  }

  private let condition = NSCondition()

  /// Performs synchronized batch-aware forward processing.
  ///
  /// - Parameters:
  ///   - tensorBatch: Input batch chunk for this worker.
  ///   - context: Batch/thread metadata used for synchronization.
  /// - Returns: Forward outputs from `BaseLayer` after synchronization.
  public override func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    if shouldPerformBatching {

      condition.lock()

      for tensor in tensorBatch {
        iterations.wrappingIncrement(ordering: .relaxed)
        performThreadBatchingForwardPass(tensor: tensor, context: context)
      }

      let sizeToCheck = if context.totalInBatch != batchSize {
        context.totalInBatch
      } else {
        batchSize
      }

      let currentIterations = iterations.load(ordering: .relaxed)

      if currentIterations == sizeToCheck {
        condition.broadcast()
      }

      while iterations.load(ordering: .relaxed) < sizeToCheck {
        condition.wait()
      }

      condition.unlock()
    }

    return super.forward(tensorBatch: tensorBatch, context: context)
  }

  /// Resets thread-batching iteration state after parameter updates.
  ///
  /// - Parameters:
  ///   - gradients: Gradient tuple forwarded to `BaseLayer`.
  ///   - learningRate: Optimizer learning rate.
  public override func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    super.apply(gradients: gradients, learningRate: learningRate)
    iterations.store(0, ordering: .relaxed)
  }

  /// Hook executed once per tensor during synchronized batch processing.
  ///
  /// Subclasses must override to update batch-shared statistics/state.
  ///
  /// - Parameters:
  ///   - tensor: Current tensor in the batch chunk.
  ///   - context: Batch/thread metadata.
  open func performThreadBatchingForwardPass(tensor: Tensor, context: NetworkContext) {
    fatalError("must override")
  }
}

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

/// Identifies the concrete type of a layer for serialization and reconstruction.
///
/// Each case maps to a distinct `Layer` subclass.  The raw string value is
/// persisted inside `.smodel` files so names must not be changed.
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
       subtract,
       divide,
       mish,
       none
}

/// A layer that performs an activation function.
public protocol ActivationLayer: Layer {
  /// The specific activation function applied by this layer.
  var type: Activation { get }
}

/// A layer that performs a convolution operation.
public protocol ConvolutionalLayer: Layer {
  /// The number of convolutional filters in this layer.
  var filterCount: Int { get }
  /// The learnable filter kernels for this layer.
  var filters: [Tensor] { get }
  /// The spatial dimensions (rows and columns) of each filter kernel.
  var filterSize: (rows: Int, columns: Int) { get }
  /// The step size used when sliding the filter over the input.
  var strides: (rows: Int, columns: Int) { get }
  /// The padding strategy applied before convolution.
  var padding: NumSwift.ConvPadding { get }
}

/// A batch of tensors used as input or output for a layer.
public typealias TensorBatch = [Tensor]

/// The primary protocol that every neural network layer must conform to.
///
/// A `Layer` object transforms an input `Tensor` into an output `Tensor` and
/// optionally maintains trainable parameters (`weights`, `biases`).  Layers are
/// chained together inside a `Sequential` container; an `Optimizer` compiles the
/// chain and drives parameter updates.
///
/// Implement `forward(tensor:context:)` with the forward math and attach a
/// `TensorContext` whose `backpropagate` closure implements the backward pass.
///
/// ## Minimal conformance
/// Inherit from `BaseLayer` instead of conforming directly — `BaseLayer` provides
/// sensible defaults for all secondary properties so you only need to override
/// `forward(tensor:context:)`, `apply(gradients:learningRate:)`, and the
/// `Codable` methods.
public protocol Layer: AnyObject, Codable {
  /// A human-readable summary describing the layer's configuration and dimensions.
  var details: String { get }
  /// The serialization identifier used when encoding and decoding this layer.
  var encodingType: EncodingType { get set }
  /// Additional encodable values that a layer may need persisted alongside standard fields.
  var extraEncodables: [String: Codable]? { get }
  /// The spatial dimensions expected at this layer's input.
  var inputSize: TensorSize { get set }
  /// The spatial dimensions produced at this layer's output.
  var outputSize: TensorSize { get }
  /// The learnable weight tensor for this layer.
  var weights: Tensor { get }
  /// The learnable bias tensor for this layer.
  var biases: Tensor { get }
  /// Whether a bias term is added during the forward pass.
  var biasEnabled: Bool { get set }
  /// Whether parameter updates are applied to this layer during training.
  var trainable: Bool { get set }
  /// Whether the layer is currently in training mode (affects dropout, batch norm, etc.).
  var isTraining: Bool { get set }
  /// The weight initializer strategy used when constructing this layer's parameters.
  var initializer: Initializer { get }
  /// The device type (CPU/GPU) to which this layer is assigned.
  var deviceType: DeviceType { get set }
  /// The compute device object used to run this layer's operations.
  var device: Device { get }
  /// Whether the optimizer manages gradient scaling before calling `apply(gradients:learningRate:)`.
  var usesOptimizer: Bool { get set }
  /// The number of samples processed simultaneously in a single forward/backward step.
  var batchSize: Int { get set }
  /// A stable string identifier used to reference this layer's output from arithmetic layers.
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
  /// The incoming weight tensors could not be applied to the layer.
  case weightImportError
  /// A generic layer error with a developer-supplied message.
  ///
  /// - Parameter error: Description of the specific problem.
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

/// An abstract base class for element-wise binary arithmetic operations between two tensor streams.
///
/// `ArithmeticLayer` walks the input tensor's computation graph to locate the output of the
/// layer identified by `linkTo`, then applies the subclass-defined `function(input:other:)`
/// to produce a combined output.  The `inverse` flag reverses the argument order.
///
/// Subclass `ArithmeticLayer` (or use the concrete `Add`, `Subtract`, `Multiply`, `Divide`)
/// to implement any custom pointwise binary operation.
open class ArithmeticLayer: BaseLayer {
  // looks up through the tensor input graph to find the first input tensor with this label applied.
  // and applies the arithmetic to it that the layer defines along with the input to this layer
  var linkTo: String
  
  private(set) var inverse: Bool
  
  override public var usesOptimizer: Bool { get { false } set { } }

  init(inputSize: TensorSize? = nil,
       initializer: InitializerType = Constants.defaultInitializer,
       biasEnabled: Bool = false,
       encodingType: EncodingType,
       inverse: Bool = false,
       linkId: String = UUID().uuidString,
       linkTo: String) {
    self.linkTo = linkTo
    self.inverse = inverse
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
               linkId: linkId,
               encodingType: encodingType)
  }

  enum CodingKeys: String, CodingKey {
    case inputSize, type, linkTo, linkId, inverse
  }
  
  /// Applies the arithmetic operation to two input tensors.
  ///
  /// Subclasses must override this method to provide the actual element-wise operation.
  ///
  /// - Parameters:
  ///   - input: The primary input tensor.
  ///   - other: The secondary input tensor (from the linked layer).
  /// - Returns: The result of applying the arithmetic operation.
  open func function(input: Tensor, other: Tensor) -> Tensor {
    fatalError("override in subclass")
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    /// do something when the input size is set when calling `compile` on `Sequential`
    /// like setting the output size or initializing the weights
    if outputSize.isEmpty {
      outputSize = inputSize
    }
  }
  
  /// Decodes an ArithmeticLayer from a serialized model.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: An error if required values cannot be decoded.
  required public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.linkTo = try container.decodeIfPresent(String.self, forKey: .linkTo) ?? ""
    self.inverse = try container.decodeIfPresent(Bool.self, forKey: .inverse) ?? false

    let linkId = try container.decodeIfPresent(String.self, forKey: .linkId) ?? UUID().uuidString
    let encodingType = try container.decodeIfPresent(EncodingType.self, forKey: .type) ?? .add

    super.init(inputSize: nil,
               biasEnabled: false,
               linkId: linkId,
               encodingType: encodingType)

    self.inputSize = try container.decodeIfPresent(TensorSize.self, forKey: .inputSize) ?? TensorSize(array: [])
  }
  
  /// Encodes the layer's properties into the given encoder.
  ///
  /// - Parameter encoder: The encoder to write data to.
  /// - Throws: An error if any values fail to encode.
  public override func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(inputSize, forKey: .inputSize)
    try container.encode(encodingType, forKey: .type)
    try container.encode(linkTo, forKey: .linkTo)
    try container.encode(linkId, forKey: .linkId)
    try container.encode(inverse, forKey: .inverse)
  }

  /// Performs the forward pass by looking up the linked input tensor and applying the binary function.
  ///
  /// - Parameters:
  ///   - tensor: The primary input tensor for the forward pass.
  ///   - context: The network context in which the forward pass is executed.
  /// - Returns: The output tensor produced by applying the layer's function to the input and linked tensors.
  public override func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    guard let other = lookupInput(input: tensor) else {
      assertionFailure("could not find reference input tensor in graph")
      return Tensor()
    }
    
    let out = if inverse {
      function(input: other, other: tensor)
    } else {
      function(input: tensor, other: other)
    }
    
    return super.forward(tensor: out, context: context)
  }
  
  func lookupInput(input: Tensor) -> Tensor? {
    return input.findInGraph(to: linkTo)
  }
}

/// A base class providing default implementations of common `Layer` properties and behaviors.
///
/// Concrete layer types should subclass `BaseLayer` and override:
/// - `forward(tensor:context:)` for the forward math
/// - `apply(gradients:learningRate:)` for parameter updates
/// - `onInputSizeSet()` to react to shape changes (e.g. initialize weights)
/// - `encode(to:)` / `init(from:)` for serialization
open class BaseLayer: Layer {
  /// A human-readable summary of the layer's input and output sizes.
  public var details: String {
    """
    Input: \(formatTensorSize(inputSize)) → Output: \(formatTensorSize(outputSize))
    """
  }

  /// The encoding type used to identify this layer during serialization.
  public var encodingType: EncodingType
  /// The input tensor size for this layer. Setting this triggers `onInputSizeSet()` to reconfigure the layer.
  public var inputSize: TensorSize = .init() {
    didSet {
      onInputSizeSet()
    }
  }
  /// The output tensor size produced by this layer.
  public var outputSize: TensorSize = .init()
  /// The learnable weight parameters of this layer.
  public var weights: Tensor = .init()
  /// The learnable bias parameters of this layer.
  public var biases: Tensor = .init()
  /// Whether bias parameters are applied during the forward pass.
  public var biasEnabled: Bool = false
  /// Whether this layer's parameters are updated during training.
  public var trainable: Bool = true
  /// Whether this layer is currently in training mode, affecting behaviors such as dropout.
  public var isTraining: Bool = true
  /// The weight initializer strategy used to initialize this layer's parameters.
  public var initializer: Initializer
  
  /// The type of device (e.g., CPU or GPU) used for computation, updating the shared DeviceManager when set.
  public var deviceType: DeviceType = .cpu {
    didSet {
      DeviceManager.shared.type = deviceType
    }
  }

  /// The compute device (e.g., CPU or GPU) used to execute this layer's operations.
  public var device: Device {
    DeviceManager.shared.device
  }

  /// The number of samples processed in a single forward/backward pass. Setting this triggers `onBatchSizeSet()`.
  public var batchSize: Int = 1 {
    didSet {
      onBatchSizeSet()
    }
  }

  /// Set this to reference the output of this layer in an arithmetic layer. eg a Shortcut path
  public var linkId: String = UUID().uuidString

  // defines whether the gradients are run through the optimizer before being applied.
  // this could be useful if a layer manages its own weight updates
  /// Whether the gradients for this layer are passed through the optimizer before being applied.
  ///
  /// Set to `false` for layers that manage their own parameter updates (e.g., `Embedding`).
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
  
  /// Not intended for direct use — concrete layer subclasses must override to implement their own deserialization.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: Always produces a default empty layer; subclasses should override.
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
    [weights]
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

/// A base class for 2-D convolutional layers, providing shared filter storage and weight management.
///
/// `BaseConvolutionalLayer` owns the `filters` array, handles filter initialization via
/// `initializeFilters()`, and consolidates `exportWeights` / `importWeights` for all
/// convolutional subclasses.  Concrete subclasses (`Conv2d`, `TransConv2d`, `DepthwiseConv2d`)
/// override `onInputSizeSet()` to derive `outputSize` and `forward(tensor:context:)` to
/// implement the specific convolution algorithm.
open class BaseConvolutionalLayer: BaseLayer, ConvolutionalLayer {
  /// A human-readable summary including filter count, strides, and padding configuration.
  public override var details: String {
    super.details +
    """
    \n
    Filters: \(filterCount)
    Strides: \(strides.rows)x\(strides.columns)
    Padding: \(padding.asString)
    """
  }

  /// A combined tensor representation of all convolutional filters, concatenated along the depth axis.
  ///
  /// Setting this property directly is not supported; use the `filters` array instead.
  public override var weights: Tensor {
    get {
      var reduce = filters
      let first = reduce.removeFirst()

      let out = reduce.reduce(first) { partialResult, new in
        partialResult.concat(new, axis: 2)
      }

      return Tensor(storage: out.storage, size: .init(rows: filterSize.rows,
                                                      columns: filterSize.columns,
                                                      depth: filterCount * inputSize.depth))

    }
    set {
      fatalError("Please use the `filters` property instead to manage weights on Convolutional layers")
    }
  }

  /// The number of convolutional filters applied at this layer.
  public var filterCount: Int
  /// The collection of filter tensors used for convolution.
  public var filters: [Tensor] = []
  /// The spatial dimensions (rows and columns) of each filter kernel.
  public var filterSize: (rows: Int, columns: Int)
  /// The step size (rows and columns) used when sliding the filter over the input.
  public var strides: (rows: Int, columns: Int)
  /// The padding strategy applied to the input before convolution.
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
  
  /// Not intended for direct use — concrete convolutional layer subclasses must override.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: Always produces a default empty layer; subclasses should override.
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

/// A base class for activation-function layers.
///
/// `BaseActivationLayer` implements the standard `forward(tensor:context:)` by
/// delegating to the device's `activate(_:_:)` / `derivate(_:_:)` methods and
/// automatically building the `TensorContext` for backpropagation.  Subclasses
/// only need to provide the `Codable` implementation and call the designated
/// initializer with the appropriate `Activation` case.
open class BaseActivationLayer: BaseLayer, ActivationLayer {

  /// Returns an empty string; activation layers have no additional detail to display.
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
  
  /// Not intended for direct use — concrete activation layer subclasses must override.
  ///
  /// - Parameter decoder: Decoder used during model loading.
  /// - Throws: Always produces a default empty activation layer; subclasses should override.
  required convenience public init(from decoder: Decoder) throws {
    self.init(inputSize: .init(),
              type: .none,
              encodingType: .none)
  }
  
  override public func onInputSizeSet() {
    super.onInputSizeSet()
    outputSize = inputSize
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
    
    let tensorContext = TensorContext { inputs, gradient, wrt in
      let derivResult = self.device.derivate(inputs, self.type)
      let outTensor = derivResult * gradient
      outTensor.label = self.type.asString() + "_input_grad"
      return (outTensor, Tensor(), Tensor())
    }
    
    let result = device.activate(tensor, type)
    let out = Tensor(storage: result.storage, size: result.size, context: tensorContext)

    out.setGraph(tensor)
    
    return super.forward(tensor: out, context: context)
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



/// A base class for normalization layers that must accumulate statistics across all batch tensors
/// before normalizing each individual tensor.
///
/// Subclasses override `performThreadBatchingForwardPass(tensor:context:)` to collect
/// per-tensor statistics under a lock, then call `super.forward(tensorBatch:context:)` once
/// all batch members have checked in.
open class BaseThreadBatchingLayer: BaseLayer {
  let updateLock = NSLock()
  let iterations = ManagedAtomic<Int>(0)

  /// Whether the layer is currently in training mode.
  ///
  /// Setting this to `false` also resets the internal iteration counter so the
  /// next training epoch starts with a clean synchronization state.
  override public var isTraining: Bool {
    didSet {
      if isTraining == false {
        iterations.store(0, ordering: .relaxed)
      }
    }
  }

  /// Whether the layer should accumulate batch statistics during the forward pass.
  ///
  /// Returns `true` by default when the layer is in training mode.
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

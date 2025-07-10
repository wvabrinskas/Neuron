//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/26/22.
//

import Foundation
import NumSwift
import NumSwiftC

/// Layer types enumeration for identifying different layer implementations
/// Used for serialization and layer identification throughout the framework
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
       none
}

/// A layer that performs an activation function
/// Activation layers apply non-linear transformations to their inputs
public protocol ActivationLayer: Layer {
  var type: Activation { get }
}

/// A layer that performs a convolution operation
/// Convolutional layers apply filters to input data using convolution operations
public protocol ConvolutionalLayer: Layer {
  var filterCount: Int { get }
  var filters: [Tensor] { get }
  var filterSize: (rows: Int, columns: Int) { get }
  var strides: (rows: Int, columns: Int) { get }
  var padding: NumSwift.ConvPadding { get }
}

/// The base protocol that all layers must implement
/// Layers are the fundamental building blocks of neural networks
public protocol Layer: AnyObject, Codable {
  /// Human readable details about the layer
  var details: String { get }
  /// The type identifier for this layer used in serialization
  var encodingType: EncodingType { get set }
  /// Additional encodable properties for custom layer implementations
  var extraEncodables: [String: Codable]? { get }
  /// The size of the input tensor expected by this layer
  var inputSize: TensorSize { get set }
  /// The size of the output tensor produced by this layer
  var outputSize: TensorSize { get }
  /// The weight parameters of the layer
  var weights: Tensor { get }
  /// The bias parameters of the layer
  var biases: Tensor { get }
  /// Whether bias terms are enabled for this layer
  var biasEnabled: Bool { get set }
  /// Whether the layer's parameters should be updated during training
  var trainable: Bool { get set }
  /// Whether the layer is currently in training mode
  var isTraining: Bool { get set }
  /// The weight initialization method used by this layer
  var initializer: Initializer? { get }
  /// The computation device (CPU/GPU) used by this layer
  var device: Device { get set }
  /// Whether gradients should be processed by the optimizer
  var usesOptimizer: Bool { get set }
  
  /// Performs the forward pass through the layer
  /// - Parameters:
  ///   - tensor: Input tensor to process
  ///   - context: Network context containing threading information
  /// - Returns: Output tensor after layer computation
  func forward(tensor: Tensor, context: NetworkContext) -> Tensor
  
  /// Applies gradients to the layer's parameters
  /// - Parameters:
  ///   - gradients: Gradients to apply
  ///   - learningRate: Learning rate for parameter updates
  func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar)
  
  /// Exports the layer's weights for serialization
  /// - Returns: Array of weight tensors
  /// - Throws: LayerErrors if weights are not initialized
  func exportWeights() throws -> [Tensor]
  
  /// Imports weights into the layer
  /// - Parameter weights: Array of weight tensors to import
  /// - Throws: LayerErrors if weights are incompatible
  func importWeights(_ weights: [Tensor]) throws
}

public enum LayerErrors: Error, LocalizedError {
  case weightImportError
  case generic(error: String)
  
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
  public var extraEncodables: [String: Codable]? {
    return [:]
  }
}

/// Base implementation of the Layer protocol
/// Provides default implementations and common functionality for all layer types
open class BaseLayer: Layer {
  public var details: String {
    """
    Input: \(formatTensorSize(inputSize)) → Output: \(formatTensorSize(outputSize))
    """
  }
  /// The type identifier for this layer used in serialization
  public var encodingType: EncodingType
  /// The size of the input tensor expected by this layer
  public var inputSize: TensorSize = .init() {
    didSet {
      onInputSizeSet()
    }
  }
  /// The size of the output tensor produced by this layer
  public var outputSize: TensorSize = .init()
  /// The weight parameters of the layer
  public var weights: Tensor = .init()
  /// The bias parameters of the layer
  public var biases: Tensor = .init()
  /// Whether bias terms are enabled for this layer
  public var biasEnabled: Bool = false
  /// Whether the layer's parameters should be updated during training
  public var trainable: Bool = true
  /// Whether the layer is currently in training mode
  public var isTraining: Bool = true
  /// The weight initialization method used by this layer
  public var initializer: Initializer?
  /// The computation device (CPU/GPU) used by this layer
  public var device: Device = CPU()
  /// Defines whether the gradients are run through the optimizer before being applied
  /// This could be useful if a layer manages its own weight updates
  public var usesOptimizer: Bool = true
  
  /// Initializes a base layer with the specified parameters
  /// - Parameters:
  ///   - inputSize: Optional input tensor size
  ///   - initializer: Weight initialization method
  ///   - biasEnabled: Whether to enable bias terms
  ///   - encodingType: Layer type identifier
  public init(inputSize: TensorSize? = nil,
              initializer: InitializerType? = nil,
              biasEnabled: Bool = false,
              encodingType: EncodingType) {
    self.inputSize = inputSize ?? TensorSize(array: [])
    self.initializer = initializer?.build()
    self.biasEnabled = biasEnabled
    self.encodingType = encodingType
    
    if inputSize != nil {
      onInputSizeSet()
    }
  }
  
  required convenience public init(from decoder: Decoder) throws {
    // override
    self.init(encodingType: .none)
  }
  
  public func encode(to encoder: Encoder) throws {
    // override
  }
  
  
  /// Performs the forward pass through the layer
  /// This is a default implementation that should be overridden by subclasses
  /// - Parameters:
  ///   - tensor: Input tensor to process
  ///   - context: Network context containing threading information
  /// - Returns: Output tensor after layer computation
  public func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    // override
    .init()
  }
  
  /// Applies gradients to the layer's parameters
  /// This is a default implementation that should be overridden by subclasses
  /// - Parameters:
  ///   - gradients: Gradients to apply
  ///   - learningRate: Learning rate for parameter updates
  public func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    // override
  }
  
  // MARK: Internal
  /// Called when the input size is set, allowing subclasses to initialize based on input dimensions
  /// Override this method to perform layer-specific initialization
  public func onInputSizeSet() {
    // override
  }
  
  public func exportWeights() throws -> [Tensor] {
    guard self.weights.isEmpty == false else {
      throw LayerErrors.generic(error: "\(encodingType.rawValue.capitalized) weights have not been initialized.")
    }
    
    return [weights]
  }
  
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
  
  private func formatTensorSize(_ size: TensorSize) -> String {
    let array = size.asArray
    if array.count <= 1 {
      return "\(array.first ?? 0)"
    }
    return array.map(String.init).joined(separator: "×")
  }
  
}

/// Base implementation for convolutional layers
/// Provides common functionality for all convolutional layer types
open class BaseConvolutionalLayer: BaseLayer, ConvolutionalLayer {
  public override var details: String {
    super.details +
    """
    \n
    Filters: \(filterCount)
    Strides: \(strides.rows)x\(strides.columns)
    Padding: \(padding.asString)
    """
  }

  /// The combined weights tensor from all filters
  /// Getting this property concatenates all individual filters
  /// Setting this property will result in a fatal error - use `filters` property instead
  public override var weights: Tensor {
    get {
      var reduce = filters
      let first = reduce.removeFirst()
      
      return reduce.reduce(first) { partialResult, new in
        partialResult.concat(new, axis: 2)
      }
    }
    set {
      fatalError("Please use the `filters` property instead to manage weights on Convolutional layers")
    }
  }
  /// Number of convolutional filters in this layer
  public var filterCount: Int
  /// Array of individual filter tensors
  public var filters: [Tensor] = []
  /// Size of each filter kernel (rows, columns)
  public var filterSize: (rows: Int, columns: Int)
  /// Stride values for convolution (rows, columns)
  public var strides: (rows: Int, columns: Int)
  /// Padding type for convolution operation
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
              encodingType: EncodingType) {
    
    self.filterCount = filterCount
    self.strides = strides
    self.padding = padding
    self.filterSize = filterSize
    
    super.init(inputSize: inputSize,
               initializer: initializer,
               biasEnabled: biasEnabled,
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
  
  public override func exportWeights() throws -> [Tensor] {
    guard filters.isEmpty == false else {
      throw LayerErrors.generic(error: "\(encodingType.rawValue.capitalized) weights have not been initialized.")
    }
    
    return filters
  }
  
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
  
  private func initializeFilters() {
    guard filters.isEmpty else {
      return
    }
    
    for _ in 0..<filterCount {
      var kernels: [[[Tensor.Scalar]]] = []
      for _ in 0..<inputSize.depth {
        var kernel: [[Tensor.Scalar]] = []
        
        for _ in 0..<filterSize.0 {
          var filterRow: [Tensor.Scalar] = []
          
          for _ in 0..<filterSize.1 {
            let weight = initializer?.calculate(input: inputSize.depth * filterSize.rows * filterSize.columns,
                                                out: inputSize.depth * filterSize.rows * filterSize.columns) ?? Tensor.Scalar.random(in: -1...1)
            filterRow.append(weight)
          }
          
          kernel.append(filterRow)
        }
        
        kernels.append(kernel)
      }
      
      let filter = Tensor(kernels)
      filters.append(filter)
    }
  }
}

open class BaseActivationLayer: BaseLayer, ActivationLayer {
  
  public override var details: String {
      ""
  }
  
  public let type: Activation

  public init(inputSize: TensorSize = TensorSize(array: []),
              type: Activation,
              encodingType: EncodingType) {
    self.type = type
    super.init(inputSize: inputSize,
               initializer: nil,
               biasEnabled: false,
               encodingType: encodingType)
    
    self.usesOptimizer = false
  }
  
  required convenience public init(from decoder: Decoder) throws {
    self.init(inputSize: .init(),
              type: .none,
              encodingType: .none)
  }
  
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    
    let context = TensorContext { inputs, gradient in
      let out = self.device.derivate(inputs, self.type).value * gradient.value
      return (Tensor(out), Tensor(), Tensor())
    }
    
    let result = device.activate(tensor, type)
    let out = Tensor(result.value, context: context)
    out.label = type.asString()

    out.setGraph(tensor)

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

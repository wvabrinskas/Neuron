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
  @discardableResult
  func forward(tensor: Tensor, context: NetworkContext) -> Tensor
  func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch
  func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar)
  func exportWeights() throws -> [Tensor]
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

open class BaseLayer: Layer {
  public var details: String {
    """
    Input: \(formatTensorSize(inputSize)) → Output: \(formatTensorSize(outputSize))
    """
  }
  
  public var encodingType: EncodingType
  public var inputSize: TensorSize = .init() {
    didSet {
      onInputSizeSet()
    }
  }
  public var outputSize: TensorSize = .init()
  public var weights: Tensor = .init()
  public var biases: Tensor = .init()
  public var biasEnabled: Bool = false
  public var trainable: Bool = true
  public var isTraining: Bool = true
  public var initializer: Initializer
  public var device: Device = CPU()
  public var batchSize: Int = 1 {
    didSet {
      onBatchSizeSet()
    }
  }
  
  // defines whether the gradients are run through the optimizer before being applied.
  // this could be useful if a layer manages its own weight updates
  public var usesOptimizer: Bool = true
  
  public init(inputSize: TensorSize? = nil,
              initializer: InitializerType = Constants.defaultInitializer,
              biasEnabled: Bool = false,
              encodingType: EncodingType) {
    self.inputSize = inputSize ?? TensorSize(array: [])
    self.initializer = initializer.build()
    self.biasEnabled = biasEnabled
    self.encodingType = encodingType
    
    if inputSize != nil {
      onInputSizeSet()
    }
  }
  
  @discardableResult
  open func callAsFunction(_ tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    forward(tensor: tensor, context: context)
  }
  
  required convenience public init(from decoder: Decoder) throws {
    // override
    self.init(encodingType: .none)
  }
  
  public func encode(to encoder: Encoder) throws {
    // override
  }
  
  public func forward(tensorBatch: TensorBatch, context: NetworkContext) -> TensorBatch {
    var result: TensorBatch = []
    
    for tensor in tensorBatch {
      result.append(forward(tensor: tensor, context: context))
    }
    
    return result
  }
  
  @discardableResult
  public func forward(tensor: Tensor, context: NetworkContext) -> Tensor {
    // override
    .init()
  }
  
  // guarenteed to be single threaded operation
  public func apply(gradients: Optimizer.Gradient, learningRate: Tensor.Scalar) {
    // override
  }
  
  // MARK: Internal
  public func onInputSizeSet() {
    // override
  }
  
  public func onBatchSizeSet() {
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
  public var filterCount: Int
  public var filters: [Tensor] = []
  public var filterSize: (rows: Int, columns: Int)
  public var strides: (rows: Int, columns: Int)
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
            let weight = initializer.calculate(input: inputSize.depth * filterSize.rows * filterSize.columns,
                                                out: inputSize.depth * filterSize.rows * filterSize.columns)
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

  public init(inputSize: TensorSize? = nil,
              type: Activation,
              encodingType: EncodingType) {
    self.type = type
    super.init(inputSize: inputSize,
               biasEnabled: false,
               encodingType: encodingType)
    
    self.usesOptimizer = false
  }
  
  required convenience public init(from decoder: Decoder) throws {
    self.init(inputSize: .init(),
              type: .none,
              encodingType: .none)
  }
  
  @discardableResult
  public override func forward(tensor: Tensor, context: NetworkContext = .init()) -> Tensor {
    
    let context = TensorContext { inputs, gradient, wrt in
      let derivResult = self.device.derivate(inputs, self.type)
      let outTensor = derivResult * gradient
      outTensor.label = self.type.asString() + "_input_grad"
      return (outTensor, Tensor(), Tensor())
    }
    
    let result = device.activate(tensor, type)
    let out = Tensor(result.storage, size: result.size, context: context)
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

  public override func apply(gradients: (weights: Tensor, biases: Tensor), learningRate: Tensor.Scalar) {
    super.apply(gradients: gradients, learningRate: learningRate)
    iterations.store(0, ordering: .relaxed)
  }

  open func performThreadBatchingForwardPass(tensor: Tensor, context: NetworkContext) {
    fatalError("must override")
  }
}

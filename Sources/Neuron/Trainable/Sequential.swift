//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import Logger

/// Metadata propagated through each layer during a forward pass, providing batch and concurrency context.
public struct NetworkContext: Sendable {
/// The global index range of samples assigned to this worker chunk within the full batch.
  public let batchRange: CountableRange<Int>
/// The position of the current sample within the active worker batch chunk.
  public let indexInBatch: Int
/// The number of samples being processed in this worker's chunk of the batch.
  public let batchProcessingCount: Int
/// The total number of samples in the full batch across all workers.
  public let totalInBatch: Int
/// A unique identifier for the thread or concurrent worker processing this batch chunk.
  public let threadId: UUID
  
  /// Creates contextual metadata for a single forward-pass invocation.
  ///
  /// This context is propagated through layers so batch-aware layers can
  /// coordinate worker-specific state and synchronization.
  ///
  /// - Parameters:
  ///   - indexInBatch: Index of the current sample within the active batch chunk.
  ///   - batchRange: Global range represented by this worker chunk.
  ///   - batchProcessingCount: Number of items processed in this worker chunk.
  ///   - totalInBatch: Total sample count in the full batch.
  ///   - threadId: Identifier used by concurrent batch workers.
  public init(indexInBatch: Int = 0,
              batchRange: CountableRange<Int> = 0..<1,
              batchProcessingCount: Int = 1,
              totalInBatch: Int = 1,
              threadId: UUID = UUID()) {
    self.indexInBatch = indexInBatch
    self.batchProcessingCount = batchProcessingCount
    self.threadId = threadId
    self.totalInBatch = totalInBatch
    self.batchRange = batchRange
  }
}

/// A sequential neural network model that chains layers in order for forward and backward passes.
public final class Sequential: Trainable, Logger {
/// Controls the verbosity of log output produced by this model.
  public var logLevel: LogLevel = .low
  
/// A human-readable identifier for this model instance.
  public var name: String = "Sequential"
/// The compute device used for inference and training; propagates to all contained layers when set.
  public var device: Device = CPU() {
    didSet {
      layers.forEach { layer in
        switch device.type {
        case .cpu:
          layer.device = CPU()
        case .gpu:
          layer.device = GPU()
        }
      }
    }
  }
  
/// Indicates whether the model is in training mode; propagates to all contained layers when set.
  public var isTraining: Bool = true {
    didSet {
      layers.forEach { $0.isTraining = isTraining }
    }
  }
  
/// The ordered collection of layers that make up the network.
  public var layers: [Layer] = []
/// Indicates whether the model has been compiled and is ready for training or inference.
  public var isCompiled: Bool = false
/// The number of samples processed per forward pass; propagates to all contained layers when set.
  public var batchSize: Int = 1 {
    didSet {
      layers.forEach { l in
        l.batchSize = batchSize
      }
    }
  }
  
  enum CodingKeys: String, CodingKey {
    case layers
  }
  
  /// Reconstructs a `Sequential` model from a serialized `.smodel` file URL.
  ///
  /// - Parameter url: File URL pointing to a previously exported model.
  /// - Returns: Decoded `Sequential` instance.
  public static func `import`(_ url: URL) -> Self {
    let result: Result<Self, Error> =  ExportHelper.buildModel(url)
    switch result {
    case .success(let model):
      return model
    case .failure(let error):
      preconditionFailure(error.localizedDescription)
    }
  }
  
  @_spi(Visualizer)
  /// Reconstructs a `Sequential` model directly from encoded model data.
  ///
  /// - Parameter data: Serialized model bytes.
  /// - Returns: Decoded `Sequential` instance.
  public static func `import`(_ data: Data) -> Self {
    let result: Result<Self, Error> =  ExportHelper.buildModel(data)
    switch result {
    case .success(let model):
      return model
    case .failure(let error):
      preconditionFailure(error.localizedDescription)
    }
  }
  
  /// Creates a sequential container from an explicit variadic layer list.
  ///
  /// - Parameter layers: Ordered layers to execute during forward passes.
  public init(_ layers: Layer...) {
    self.layers = layers
  }
  
  /// Creates a sequential container from a layer builder closure.
  ///
  /// - Parameter layers: Closure returning layers in execution order.
  public init(_ layers: () -> [Layer]) {
    self.layers = layers()
  }
  
  convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    
    var layers: [Layer] = []
    let layerModels = try container.decodeIfPresent([LayerModel].self, forKey: .layers)
    layerModels?.forEach({ model in
      layers.append(model.layer)
    })
    
    self.init({ layers })
  }
  
  /// Runs inference on a single tensor using call syntax.
  ///
  /// - Parameters:
  ///   - data: Input tensor.
  ///   - context: Batch/thread metadata propagated through the network.
  /// - Returns: Final network output tensor.
  public func callAsFunction(_ data: Tensor, context: NetworkContext) -> Tensor {
    predict(data, context: context)
  }
  
  /// Runs inference on a batch of tensors using call syntax.
  ///
  /// - Parameters:
  ///   - data: Input tensor batch.
  ///   - context: Batch/thread metadata propagated through the network.
  /// - Returns: Output tensor batch in input order.
  public func callAsFunction(_ data: TensorBatch, context: NetworkContext) -> TensorBatch {
    predict(batch: data, context: context)
  }
  
  /// Encodes this network and its layers for persistence.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(layers.map { LayerModel(layer: $0) }, forKey: .layers)
  }
  
  /// Applies one set of accumulated layer gradients to all layers.
  ///
  /// - Parameters:
  ///   - gradients: Gradient payload where each index maps to a layer.
  ///   - learningRate: Scalar step size used by layer update rules.
  public func apply(gradients: Tensor.Gradient, learningRate: Tensor.Scalar) {
    for i in 0..<layers.count {
      let layer = layers[i]
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
      layer.apply(gradients: (weights: gradient, biases: biasGradient), learningRate: learningRate)
    }
  }
  
  /// Performs a full forward pass for a batch.
  ///
  /// - Parameters:
  ///   - batch: Input tensors to process in order.
  ///   - context: Batch/thread metadata for downstream layers.
  /// - Returns: Output tensors produced by the last layer.
  public func predict(batch: TensorBatch, context: NetworkContext) -> TensorBatch {
    precondition(isCompiled, "Please call compile() on the \(self) before attempting to fit")
    
    var outputTensors = batch
    
    layers.forEach { layer in
      let newTensors = layer.forward(tensorBatch: outputTensors, context: context)

      for (i, tensor) in newTensors.enumerated() {
        tensor.label = layer.encodingType.rawValue

        if tensor.graph[outputTensors[i].id] == nil {
          tensor.setGraph(outputTensors[i])
        }
      }

      outputTensors = newTensors
    }
    
    return outputTensors
  }
  
  /// Performs a full forward pass for one tensor.
  ///
  /// - Parameters:
  ///   - data: Input tensor.
  ///   - context: Batch/thread metadata for downstream layers.
  /// - Returns: Output tensor produced by the last layer.
  public func predict(_ data: Tensor, context: NetworkContext) -> Tensor {
    precondition(isCompiled, "Please call compile() on the \(self) before attempting to fit")
    
    var outputTensor = data
    
    layers.forEach { layer in
      let newTensor = layer.forward(tensor: outputTensor, context: context)
      newTensor.label = layer.encodingType.rawValue
      newTensor.setGraph(outputTensor)
      outputTensor = newTensor
    }
    
    return outputTensor
  }
  
  /// Validates layer connectivity and propagates inferred input sizes.
  ///
  /// The first layer must have an explicit `inputSize`; subsequent layers
  /// receive their input sizes from the previous layer's output shape.
  public func compile() {
    var inputSize: TensorSize = TensorSize(array: [])
    var i = 0
    
    var errorMsg: String = ""
    
    for layer in layers {
      if i == 0 && layer.inputSize.isEmpty {
        fatalError("The first layer should contain an input size")
      }

      if i > 0 {
        
        if inputSize.columns == 0 {
          errorMsg = "inputSize.columns of \(layer.self) cannot be 0"
          break
        } else if inputSize.rows == 0 {
          errorMsg = "inputSize.rows of \(layer.self) cannot be 0"
          break
        } else if inputSize.depth == 0 {
          errorMsg = "inputSize.depth of \(layer.self) cannot be 0"
          break
        }
        
        layer.inputSize = inputSize
      }
      
      inputSize = layer.outputSize
      i += 1
    }
    
    if errorMsg.isEmpty == false {
      log(type: .error, priority: .alwaysShow, message: errorMsg)
      isCompiled = false
      return
    }
    
    isCompiled = true
  }
  
  /// Exports all layer weights in layer order.
  ///
  /// - Returns: Nested weight tensor collection per layer.
  /// - Throws: `LayerErrors` if the network has not been compiled.
  public func exportWeights() throws -> [[Tensor]] {
    guard isCompiled else {
      throw LayerErrors.generic(error: "Please compile the trainable first before attempting to export weights.")
    }
    
    return try layers.map { try $0.exportWeights() }
  }
  
  /// Imports precomputed weights for each layer.
  ///
  /// - Parameter weights: Nested tensor list where each entry maps to a layer.
  /// - Throws: `LayerErrors` when the network is not compiled or shapes mismatch.
  public func importWeights(_ weights: [[Tensor]]) throws {
    guard isCompiled else {
      throw LayerErrors.generic(error: "Please compile the trainable first before attempting to import weights.")
    }
    
    for i in 0..<layers.count {
      let layer = layers[i]
      let weights = weights[i]
      try layer.importWeights(weights)
    }
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import Logger

/// Context information passed through the network during computation
/// Contains threading and execution context data
public struct NetworkContext: Sendable {
  /// Thread identifier for parallel processing
  public var threadId: Int
  
  /// Initializes network context with thread information
  /// - Parameter threadId: Thread identifier for parallel processing
  public init(threadId: Int = 0) {
    self.threadId = threadId
  }
}

/// Sequential neural network model that chains layers together
/// Organizes layers in a linear sequence for forward and backward passes
public final class Sequential: Trainable, Logger {
  /// Logging level for debug output
  public var logLevel: LogLevel = .low
  
  /// Name identifier for this sequential model
  public var name: String = "Sequential"
  /// Computation device for all layers in the model
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
  
  /// Training mode flag that propagates to all layers
  public var isTraining: Bool = true {
    didSet {
      layers.forEach { $0.isTraining = isTraining }
    }
  }
  
  /// Array of layers in sequential order
  public var layers: [Layer] = []
  /// Whether the model has been compiled and is ready for training/inference
  public var isCompiled: Bool = false
  
  enum CodingKeys: String, CodingKey {
    case layers
  }
  
  /// Imports a Sequential model from a saved file
  /// - Parameter url: URL to the saved model file
  /// - Returns: The loaded Sequential model
  public static func `import`(_ url: URL) -> Self {
    let result: Result<Self, Error> =  ExportHelper.buildModel(url)
    switch result {
    case .success(let model):
      return model
    case .failure(let error):
      preconditionFailure(error.localizedDescription)
    }
  }
  
  /// Initializes a Sequential model with variadic layers
  /// - Parameter layers: Variable number of layers to add to the model
  public init(_ layers: Layer...) {
    self.layers = layers
  }
  
  /// Initializes a Sequential model with a closure returning layers
  /// - Parameter layers: Closure that returns an array of layers
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
  
  /// Allows the Sequential model to be called as a function
  /// - Parameters:
  ///   - data: Input tensor to process
  ///   - context: Network context for computation
  /// - Returns: Output tensor after processing through all layers
  public func callAsFunction(_ data: Tensor, context: NetworkContext) -> Tensor {
    predict(data, context: context)
  }
  
  /// Encodes the Sequential model for serialization
  /// - Parameter encoder: Encoder to use for serialization
  /// - Throws: Encoding errors if serialization fails
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(layers.map { LayerModel(layer: $0) }, forKey: .layers)
  }
  
  /// Makes predictions using the Sequential model
  /// - Parameters:
  ///   - data: Input tensor to process
  ///   - context: Network context for computation
  /// - Returns: Output tensor after processing through all layers
  public func predict(_ data: Tensor, context: NetworkContext) -> Tensor {
    precondition(isCompiled, "Please call compile() on the \(self) before attempting to fit")
    
    var outputTensor = data
    
    layers.forEach { layer in
      let newTensor = layer.forward(tensor: outputTensor, context: context)
      if newTensor.graph == nil {
        newTensor.setGraph(outputTensor)
      }
      outputTensor = newTensor
    }
    
    return outputTensor
  }
  
  /// Compiles the Sequential model by calculating layer dimensions and validating the architecture
  /// This method must be called before training or inference
  /// The first layer must have an input size specified
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
  
  public func exportWeights() throws -> [[Tensor]] {
    guard isCompiled else {
      throw LayerErrors.generic(error: "Please compile the trainable first before attempting to export weights.")
    }
    
    return try layers.map { try $0.exportWeights() }
  }
  
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

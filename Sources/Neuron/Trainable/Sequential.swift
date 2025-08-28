//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import Logger

public struct NetworkContext: Sendable {
  public let batchRange: CountableRange<Int>
  public let indexInBatch: Int
  public let batchProcessingCount: Int
  public let totalInBatch: Int
  public let threadId: UUID
  
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

public final class Sequential: Trainable, Logger {
  public var logLevel: LogLevel = .low
  
  public var name: String = "Sequential"
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
  
  public var isTraining: Bool = true {
    didSet {
      layers.forEach { $0.isTraining = isTraining }
    }
  }
  
  public var layers: [Layer] = []
  public var isCompiled: Bool = false
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
  public static func `import`(_ data: Data) -> Self {
    let result: Result<Self, Error> =  ExportHelper.buildModel(data)
    switch result {
    case .success(let model):
      return model
    case .failure(let error):
      preconditionFailure(error.localizedDescription)
    }
  }
  
  public init(_ layers: Layer...) {
    self.layers = layers
  }
  
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
  
  public func callAsFunction(_ data: Tensor, context: NetworkContext) -> Tensor {
    predict(data, context: context)
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(layers.map { LayerModel(layer: $0) }, forKey: .layers)
  }
  
  public func apply(gradients: Tensor.Gradient, learningRate: Float) {
    for i in 0..<layers.count {
      let layer = layers[i]
      let gradient = gradients.weights[i]
      let biasGradient = gradients.biases[i]
      layer.apply(gradients: (weights: gradient, biases: biasGradient), learningRate: learningRate)
    }
  }
  
  public func predict(batch: TensorBatch, context: NetworkContext) -> TensorBatch {
    precondition(isCompiled, "Please call compile() on the \(self) before attempting to fit")
    
    var outputTensors = batch
    
    layers.forEach { layer in
      let newTensors = layer.forward(tensorBatch: outputTensors, context: context)
      
      for (i, tensor) in newTensors.enumerated() {
        if tensor.graph[outputTensors[i].id] == nil {
          tensor.setGraph(outputTensors[i])
        }
      }

      outputTensors = newTensors
    }
    
    return outputTensors
  }
  
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

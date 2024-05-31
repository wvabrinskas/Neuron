//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation

public final class Sequential: Trainable {
  public var threadId: Int = 0 {
    didSet {
      layers.forEach { $0.threadId = threadId }
    }
  }
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
  
  public func callAsFunction(_ data: Tensor) -> Tensor {
    predict(data)
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(layers.map { LayerModel(layer: $0) }, forKey: .layers)
  }
  
  public func predict(_ data: Tensor) -> Tensor {
    precondition(isCompiled, "Please call compile() on the \(self) before attempting to fit")
    
    var outputTensor = data
    
    layers.forEach { layer in
      let newTensor = layer.forward(tensor: outputTensor)
      if newTensor.graph == nil {
        newTensor.setGraph(outputTensor)
      }
      outputTensor = newTensor
    }
    
    return outputTensor
  }
  
  public func compile() {
    var inputSize: TensorSize = TensorSize(array: [])
    var i = 0
    layers.forEach { layer in
      if i == 0 && layer.inputSize.isEmpty {
        fatalError("The first layer should contain an input size")
      }
      
      if i > 0 {
        layer.inputSize = inputSize
      }
      
      inputSize = layer.outputSize
      i += 1
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

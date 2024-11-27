//
//  BaseTrainable.swift
//  Neuron
//
//  Created by William Vabrinskas on 11/26/24.
//

import Foundation

open class BaseTrainable: Trainable {
  open var name: String = "Sequential"
  open var device: Device = CPU() {
    didSet {
      setDevice()
    }
  }
  
  open var isTraining: Bool = true {
    didSet {
      setTrainingMode()
    }
  }
  
  open var layers: [Layer] = []
  open var isCompiled: Bool = false
  
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
  
  required convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    
    var layers: [Layer] = []
    let layerModels = try container.decodeIfPresent([LayerModel].self, forKey: .layers)
    layerModels?.forEach({ model in
      layers.append(model.layer)
    })
    
    self.init({ layers })
  }
  
  open func callAsFunction(_ data: Tensor, context: NetworkContext) -> Tensor {
    predict(data, context: context)
  }
  
  open func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(layers.map { LayerModel(layer: $0) }, forKey: .layers)
  }
  
  open func predict(_ data: Tensor, context: NetworkContext) -> Tensor {
    fatalError("please use one of the premade trainables")
  }
  
  open func compile() {
    fatalError("please use one of the premade trainables")
  }
  
  open func exportWeights() throws -> [[Tensor]] {
    guard isCompiled else {
      throw LayerErrors.generic(error: "Please compile the trainable first before attempting to export weights.")
    }
    
    return try layers.map { try $0.exportWeights() }
  }
  
  open func importWeights(_ weights: [[Tensor]]) throws {
    guard isCompiled else {
      throw LayerErrors.generic(error: "Please compile the trainable first before attempting to import weights.")
    }
    
    for i in 0..<layers.count {
      let layer = layers[i]
      let weights = weights[i]
      try layer.importWeights(weights)
    }
  }
  
  open func setDevice() {
    layers.forEach { layer in
      switch device.type {
      case .cpu:
        layer.device = CPU()
      case .gpu:
        layer.device = GPU()
      }
    }
  }
  
  open func setTrainingMode() {
    layers.forEach { $0.isTraining = isTraining }
  }
}

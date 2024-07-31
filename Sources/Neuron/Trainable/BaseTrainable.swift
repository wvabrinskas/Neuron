//
//  BaseTrainable.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/30/24.
//

import Foundation

open class BaseTrainable<N: TensorNumeric>: Trainable {
  open var threadId: Int = 0 {
    didSet {
      layers.forEach { $0.threadId = threadId }
    }
  }
  open var device: Device = CPU() {
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
  
  open var isTraining: Bool = true {
    didSet {
      layers.forEach { $0.isTraining = isTraining }
    }
  }
  
  public var name: String = "base"
  public var layers: [BaseLayer<N>] = []
  public var isCompiled: Bool = false
  
  enum CodingKeys: String, CodingKey {
    case layers
  }
  
  public init(_ layers: BaseLayer<N>...) {
    self.layers = layers
  }
  
  public init(_ layers: () -> [BaseLayer<N>]) {
    self.layers = layers()
  }
  
  required convenience public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    
    var layers: [BaseLayer<N>] = []
    let layerModels = try container.decodeIfPresent([LayerModel<N>].self, forKey: .layers)
    layerModels?.forEach({ model in
      layers.append(model.layer)
    })
    
    self.init({ layers })
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
  
  public func callAsFunction(_ data: Tensor) -> Tensor {
    predict(data)
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(layers.map { LayerModel(layer: $0) }, forKey: .layers)
  }
  
  open func predict(_ data: Tensor) -> Tensor {
    .init()
  }
  
  open func compile() {
    isCompiled = true
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
  
}

//
//  Node.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import SwiftUI
@_spi(Visualizer) import Neuron
import NumSwift


struct NodePayload {
  var layer: EncodingType
  var outputSize: TensorSize
  var inputSize: TensorSize
  var parameters: Int
  var details: String
  var layerType: BaseLayerType
  
  init(layer: EncodingType,
       outputSize: TensorSize,
       inputSize: TensorSize,
       parameters: Int,
       details: String,
       layerType: BaseLayerType = .regular) {
    self.layer = layer
    self.outputSize = outputSize
    self.inputSize = inputSize
    self.parameters = parameters
    self.details = details
    self.layerType = layerType
  }
  
  init(layer: Layer) {
    self.layer = layer.encodingType
    self.outputSize = layer.outputSize
    self.inputSize = layer.inputSize
    self.parameters = layer.weights.shape.reduce(1, *)
    self.details = layer.details
    
    layerType = if layer is ActivationLayer {
      .activation
    } else {
      .regular
    }
  }
}

protocol Node: AnyObject {
  var parentPoint: CGPoint? { get set }
  var point: CGPoint? { get set }
  var connections: [Node] { get set }
  init(payload: NodePayload)
  
  @ViewBuilder
  func build() -> any View
}

class BaseNode: Node {
  var parentPoint: CGPoint?
  var point: CGPoint?
  
  var connections: [Node] = []
  let layer: EncodingType
  let payload: NodePayload
  
  func build() -> any View {
    EmptyView()
  }

  required init(payload: NodePayload = .init(layer: .none,
                                             outputSize: .init(array: []),
                                             inputSize: .init(array: []),
                                             parameters: 0,
                                             details: "")) {
    self.payload = payload
    self.layer = payload.layer
  }
  
}

enum BaseLayerType {
  case regular, activation
}

extension EncodingType {
  var color: Color {
    switch self {
    case .leakyRelu, .relu, .sigmoid, .tanh, .swish, .selu, .softmax: Color(red: 0.2, green: 0.6, blue: 0.2)
    case .batchNormalize, .layerNormalize: Color(red: 0.6, green: 0.5, blue: 0.8)
    case .conv2d, .transConv2d: Color(red: 0, green: 0.5, blue: 0.8)
    case .dense: Color(red: 0.8, green: 0.4, blue: 0.4)
    case .dropout: Color(red: 0.7, green: 0.1, blue: 0.7)
    case .flatten: Color(red: 0.7, green: 0.4, blue: 0.8)
    case .maxPool, .avgPool: Color(red: 0.9, green: 0.6, blue: 0.2)
    case .reshape: Color(red: 0.8, green: 0.6, blue: 0.8)
    case .lstm, .embedding: Color(red: 0.6, green: 0.4, blue: 0.8)
      default: Color(red: 0.5, green: 0.5, blue: 0.5)
    }
  }
  
}

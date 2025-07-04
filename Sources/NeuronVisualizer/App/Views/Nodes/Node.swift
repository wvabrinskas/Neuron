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
  
  init(layer: EncodingType,
       outputSize: TensorSize,
       inputSize: TensorSize,
       parameters: Int,
       details: String) {
    self.layer = layer
    self.outputSize = outputSize
    self.inputSize = inputSize
    self.parameters = parameters
    self.details = details
  }
  
  init(layer: Layer) {
    self.layer = layer.encodingType
    self.outputSize = layer.outputSize
    self.inputSize = layer.inputSize
    self.parameters = layer.weights.shape.reduce(1, *)
    self.details = layer.details
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
  let layer: VisualEncodingType
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
    self.layer = VisualEncodingType(encodingType: payload.layer)
  }
  
}

@available(macOS 14, *)
public enum VisualEncodingType: String, Codable {
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
  
  var color: Color {
    switch self {
    case .leakyRelu, .relu, .sigmoid, .tanh, .swish, .selu, .softmax: Color(red: 0.2, green: 0.6, blue: 0.2)
    case .batchNormalize, .layerNormalize: Color(red: 0.3, green: 0.5, blue: 0.8)
    case .conv2d, .transConv2d: Color(red: 0.3, green: 0.5, blue: 0.8)
    case .dense: Color(red: 0.8, green: 0.4, blue: 0.4)
    case .dropout: Color(red: 0.7, green: 0.7, blue: 0.7)
    case .flatten: Color(red: 0.7, green: 0.4, blue: 0.8)
    case .maxPool, .avgPool: Color(red: 0.9, green: 0.6, blue: 0.2)
    case .reshape: Color(red: 0.8, green: 0.6, blue: 0.8)
    case .lstm, .embedding: Color(red: 0.6, green: 0.4, blue: 0.8)
      default: Color(red: 0.5, green: 0.5, blue: 0.5)
    }
  }
  
  init(encodingType: EncodingType) {
    switch encodingType {
    case .leakyRelu: self = .leakyRelu
    case .relu: self = .relu
    case .sigmoid: self = .sigmoid
    case .softmax: self = .softmax
    case .swish: self = .swish
    case .tanh: self = .tanh
    case .batchNormalize: self = .batchNormalize
    case .conv2d: self = .conv2d
    case .dense: self = .dense
    case .dropout: self = .dropout
    case .flatten: self = .flatten
    case .maxPool: self = .maxPool
    case .reshape: self = .reshape
    case .transConv2d: self = .transConv2d
    case .layerNormalize: self = .layerNormalize
    case .lstm: self = .lstm
    case .embedding: self = .embedding
    case .avgPool: self = .avgPool
    case .selu: self = .selu
    case .none: self = .none
    }
  }
}

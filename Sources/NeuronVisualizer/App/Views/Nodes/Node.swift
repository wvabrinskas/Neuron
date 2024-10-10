//
//  Node.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import SwiftUI
import Neuron


struct NodePayload {
  var layer: EncodingType
  var outputSize: TensorSize
  var inputSize: TensorSize
  
  init(layer: EncodingType, outputSize: TensorSize, inputSize: TensorSize) {
    self.layer = layer
    self.outputSize = outputSize
    self.inputSize = inputSize
  }
  
  init(layer: Layer) {
    self.layer = layer.encodingType
    self.outputSize = layer.outputSize
    self.inputSize = layer.inputSize
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

@available(macOS 14, *)
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
                                             inputSize: .init(array: []))) {
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
    case .leakyRelu, .relu, .sigmoid, .tanh, .swish, .selu, .softmax: .green
    case .batchNormalize, .layerNormalize: .blue
    case .conv2d, .transConv2d: .cyan
    case .dense: .indigo
    case .dropout: .red
    case .flatten: .orange
    case .maxPool, .avgPool: .brown
    case .reshape: .pink
    case .lstm, .embedding: .purple
      default: .gray
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

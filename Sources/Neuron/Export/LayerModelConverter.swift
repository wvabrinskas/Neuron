//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/31/22.
//

import Foundation
import NumSwift

public struct LayerModelConverter<N: TensorNumeric> {
  public enum CodingKeys: String, CodingKey {
    case layer
  }
  
  static func convert(decoder: Decoder, type: EncodingType) throws -> (BaseLayer<N>)? {
    let con = try decoder.container(keyedBy: CodingKeys.self)
    var layer: (BaseLayer<N>)?
    
    switch type {
    case .leakyRelu:
      layer = try getLayer(layer: LeakyReLu<N>.self, container: con)
    case .relu:
      layer = try getLayer(layer: ReLu<N>.self, container: con)
    case .sigmoid:
      layer = try getLayer(layer: Sigmoid<N>.self, container: con)
    case .softmax:
      layer = try getLayer(layer: Softmax<N>.self, container: con)
    case .swish:
      layer = try getLayer(layer: Swish<N>.self, container: con)
    case .tanh:
      layer = try getLayer(layer: Tanh<N>.self, container: con)
    case .batchNormalize:
      layer = try getLayer(layer: BatchNormalize<N>.self, container: con)
    case .conv2d:
      layer = try getLayer(layer: Conv2d<N>.self, container: con)
    case .dense:
      layer = try getLayer(layer: Dense<N>.self, container: con)
    case .dropout:
      layer = try getLayer(layer: Dropout<N>.self, container: con)
    case .flatten:
      layer = try getLayer(layer: Flatten<N>.self, container: con)
    case .maxPool:
      layer = try getLayer(layer: MaxPool<N>.self, container: con)
    case .reshape:
      layer = try getLayer(layer: Reshape<N>.self, container: con)
    case .transConv2d:
      layer = try getLayer(layer: TransConv2d<N>.self, container: con)
    case .layerNormalize:
      layer = try getLayer(layer: LayerNormalize<N>.self, container: con)
    case .lstm:
      layer = try getLayer(layer: LSTM<N>.self, container: con)
    case .embedding:
      layer = try getLayer(layer: Embedding<N>.self, container: con)
    case .avgPool:
      layer = try getLayer(layer: AvgPool<N>.self, container: con)
    case .selu:
      layer = try getLayer(layer: SeLu<N>.self, container: con)
    case .none:
      layer = nil
    }

    return layer
  }
  
  private static func getLayer<L: BaseLayer<N>>(layer: L.Type,
                                         container: KeyedDecodingContainer<CodingKeys>) throws -> (BaseLayer<N>)? {
    
    if let cont = try container.decodeIfPresent([L].self, forKey: .layer),
       let arrLayer = cont.first {
      return arrLayer
    }
    
    return nil
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/31/22.
//

import Foundation
import NumSwift

public struct LayerModelConverter<T: TensorNumeric> {
  public enum CodingKeys: String, CodingKey {
    case layer
  }
  
  static func convert(decoder: Decoder, type: EncodingType) throws -> (BaseLayer<T>)? {
    let con = try decoder.container(keyedBy: CodingKeys.self)
    var layer: (BaseLayer<T>)?
    
    switch type {
    case .leakyRelu:
      layer = try getLayer(layer: LeakyReLu<T>.self, container: con)
    case .relu:
      layer = try getLayer(layer: ReLu<T>.self, container: con)
    case .sigmoid:
      layer = try getLayer(layer: Sigmoid<T>.self, container: con)
    case .softmax:
      layer = try getLayer(layer: Softmax<T>.self, container: con)
    case .swish:
      layer = try getLayer(layer: Swish<T>.self, container: con)
    case .tanh:
      layer = try getLayer(layer: Tanh<T>.self, container: con)
    case .batchNormalize:
      layer = try getLayer(layer: BatchNormalize<T>.self, container: con)
    case .conv2d:
      layer = try getLayer(layer: Conv2d<T>.self, container: con)
    case .dense:
      layer = try getLayer(layer: Dense<T>.self, container: con)
    case .dropout:
      layer = try getLayer(layer: Dropout<T>.self, container: con)
    case .flatten:
      layer = try getLayer(layer: Flatten<T>.self, container: con)
    case .maxPool:
      layer = try getLayer(layer: MaxPool<T>.self, container: con)
    case .reshape:
      layer = try getLayer(layer: Reshape<T>.self, container: con)
    case .transConv2d:
      layer = try getLayer(layer: TransConv2d<T>.self, container: con)
    case .layerNormalize:
      layer = try getLayer(layer: LayerNormalize<T>.self, container: con)
    case .lstm:
      layer = try getLayer(layer: LSTM<T>.self, container: con)
    case .embedding:
      layer = try getLayer(layer: Embedding<T>.self, container: con)
    case .avgPool:
      layer = try getLayer(layer: AvgPool<T>.self, container: con)
    case .selu:
      layer = try getLayer(layer: SeLu<T>.self, container: con)
    case .none:
      layer = nil
    }

    return layer
  }
  
  private static func getLayer<L: BaseLayer<T>>(layer: L.Type,
                                         container: KeyedDecodingContainer<CodingKeys>) throws -> (BaseLayer<T>)? {
    
    if let cont = try container.decodeIfPresent([L].self, forKey: .layer),
       let arrLayer = cont.first {
      return arrLayer
    }
    
    return nil
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/31/22.
//

import Foundation
import NumSwift

public struct LayerModelConverter {
  public enum CodingKeys: String, CodingKey {
    case layer
  }
  
  static func convert(decoder: Decoder, type: EncodingType) throws -> Layer? {
    let con = try decoder.container(keyedBy: CodingKeys.self)
    var layer: Layer?
    
    switch type {
    case .leakyRelu:
      layer = try getLayer(layer: LeakyReLu.self, container: con)
    case .relu:
      layer = try getLayer(layer: ReLu.self, container: con)
    case .sigmoid:
      layer = try getLayer(layer: Sigmoid.self, container: con)
    case .softmax:
      layer = try getLayer(layer: Softmax.self, container: con)
    case .swish:
      layer = try getLayer(layer: Swish.self, container: con)
    case .tanh:
      layer = try getLayer(layer: Tanh.self, container: con)
    case .batchNormalize:
      layer = try getLayer(layer: BatchNormalize.self, container: con)
    case .conv2d:
      layer = try getLayer(layer: Conv2d.self, container: con)
    case .dense:
      layer = try getLayer(layer: Dense.self, container: con)
    case .dropout:
      layer = try getLayer(layer: Dropout.self, container: con)
    case .flatten:
      layer = try getLayer(layer: Flatten.self, container: con)
    case .maxPool:
      layer = try getLayer(layer: MaxPool.self, container: con)
    case .reshape:
      layer = try getLayer(layer: Reshape.self, container: con)
    case .transConv2d:
      layer = try getLayer(layer: TransConv2d.self, container: con)
    case .layerNormalize:
      layer = try getLayer(layer: LayerNormalize.self, container: con)
    case .lstm:
      layer = try getLayer(layer: LSTM.self, container: con)
    case .embedding:
      layer = try getLayer(layer: Embedding.self, container: con)
    case .none:
      layer = nil
    }

    return layer
  }
  
  private static func getLayer<T: Layer>(layer: T.Type,
                                         container: KeyedDecodingContainer<CodingKeys>) throws -> Layer? {
    
    if let cont = try container.decodeIfPresent([T].self, forKey: .layer),
       let arrLayer = cont.first {
      return arrLayer
    }
    
    return nil
  }
}

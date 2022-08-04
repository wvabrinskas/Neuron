//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/31/22.
//

import Foundation
import NumSwift

public class LayerModel: Codable {
  var layer: Layer
  
  init(layer: Layer) {
    self.layer = layer
  }
  
  public enum CodingKeys: String, CodingKey {
    case layer, type
  }
  
  public required init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    
    guard let type = try container.decodeIfPresent(EncodingType.self, forKey: .type) else {
      preconditionFailure("no type specified")
    }
    
    if let layer = try LayerModelConverter.convert(decoder: decoder, type: type) {
      self.layer = layer
      return
    }
    
    preconditionFailure("Could not convert layer")
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    var nested = container.nestedUnkeyedContainer(forKey: .layer)
    try layer.encode(to: nested.superEncoder())
    try container.encode(layer.encodingType, forKey: .type)
  }
}


extension NumSwift.ConvPadding: Codable {
  private enum CodingKeys: String, CodingKey {
    case same
    case valid
  }
  
  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    
    if let _ = try values.decodeIfPresent(String.self, forKey: .valid) {
      self = .valid
      return
    }
    
    if let _ = try values.decodeIfPresent(String.self, forKey: .same) {
      self = .same
      return
    }
    
    self = .same
  }
  
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    switch self {
    case .same:
      try container.encode(CodingKeys.same.rawValue, forKey: .same)
    case .valid:
      try container.encode(CodingKeys.valid.rawValue, forKey: .valid)
    }
  }
}

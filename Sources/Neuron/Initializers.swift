//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import GameplayKit

/// Weight initializer methods
public enum InitializerType: Codable {
  ///Generates weights based on a normal gaussian distribution. Mean = 0 sd = 1
  case xavierNormal
  ///Generates weights based on a uniform distribution
  case xavierUniform
  
  case heNormal
  case heUniform
  case normal(std: Float)
  
  public func build() -> Initializer {
    Initializer(type: self)
  }
}

public struct Initializer {
  public let type: InitializerType
  private var dist: NormalDistribution = NormalDistribution(mean: 0, deviation: 1)
  
  public enum CodingKeys: String, CodingKey, CaseIterable {
    case xavierNormal
    case xavierUniform
    case heNormal
    case heUniform
    case normal
    
    var type: InitializerType {
      switch self {
      case .xavierNormal:
        return .xavierNormal
      case .xavierUniform:
        return .xavierUniform
      case .heNormal:
        return .heNormal
      case .heUniform:
        return .heUniform
      case .normal:
        return .normal(std: 1)
      }
    }
  }
  
  public init(type: InitializerType) {
    self.type = type
    switch type {
    case .normal(let std):
      self.dist = NormalDistribution(mean: 0, deviation: std)
    default:
      break
    }
  }
  
  public func calculate(input: Int, out: Int = 0) -> Float {
    switch type {
      
    case .xavierUniform:
      let min = -Float(sqrt(6) / sqrt((Double(input) + Double(out))))
      let max = Float(sqrt(6) / sqrt((Double(input) + Double(out))))
      
      return Float.random(in: min...max)
      
    case .xavierNormal:
      return dist.nextFloat() * Float(sqrt(2 / (Double(input) + Double(out))))
      
    case .heUniform:
      let min = -Float(sqrt(6) / sqrt((Double(input))))
      let max = Float(sqrt(6) / sqrt((Double(input))))
      
      return Float.random(in: min...max)
      
    case .heNormal:
      return dist.nextFloat() * Float(sqrt(2 / (Double(input))))
      
    case .normal:
      return dist.nextFloat()
    }
    
  }
}

extension Initializer: Codable {
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: Self.CodingKeys)
    
    switch self.type {
    case .xavierUniform:
      try container.encode(CodingKeys.xavierUniform.stringValue, forKey: .xavierUniform)
    case .xavierNormal:
      try container.encode(CodingKeys.xavierNormal.stringValue, forKey: .xavierUniform)
    case .heUniform:
      try container.encode(CodingKeys.heUniform.stringValue, forKey: .xavierUniform)
    case .heNormal:
      try container.encode(CodingKeys.heNormal.stringValue, forKey: .xavierUniform)
    case .normal(let std):
      try container.encode(CodingKeys.normal.stringValue + "-\(std)", forKey: .xavierUniform)
    }
  }
  
  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    self = .init(type: .heUniform)

    try values.allKeys.forEach { k in
      if k == .normal {
        if let val = try values.decodeIfPresent(String.self, forKey: .normal) {
          let split = val.split(separator: "-")[safe: 1, ""]
          if let std = Float(split) {
            self = .init(type: .normal(std: std))
          } else {
            self = .init(type: .normal(std: 1))
          }
        } else {
          self = .init(type: .normal(std: 1))
        }
      } else {
        self = .init(type: k.type)
      }
    }
  }
}

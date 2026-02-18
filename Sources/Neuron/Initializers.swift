//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import GameplayKit

/// Weight initializer methods
public enum InitializerType: Codable, Equatable {
  ///Generates weights based on a normal gaussian distribution. Mean = 0 sd = 1
  case xavierNormal
  ///Generates weights based on a uniform distribution
  case xavierUniform
  
  case heNormal
  case heUniform
  case normal(std: Tensor.Scalar)
  
  /// Creates an executable initializer from this initializer type.
  ///
  /// - Returns: `Initializer` wrapper configured for this case.
  public func build() -> Initializer {
    Initializer(type: self)
  }
}

public struct Initializer {
  public let type: InitializerType
  private var dist: Gaussian = Gaussian(std: 1, mean: 0)
  
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
  
  /// Creates an initializer strategy wrapper.
  ///
  /// - Parameter type: Weight initialization strategy.
  public init(type: InitializerType) {
    self.type = type
    switch type {
    case .normal(let std):
      self.dist = Gaussian(std: Double(std), mean: 0)
    default:
      break
    }
  }
  
  /// Generates one initialized scalar value.
  ///
  /// - Parameters:
  ///   - input: Fan-in size.
  ///   - out: Fan-out size (used by Xavier variants).
  /// - Returns: One initialized scalar sampled using the selected strategy.
  public func calculate(input: Int, out: Int = 0) -> Tensor.Scalar {
    switch type {
    
    case .xavierUniform:
      let value = Tensor.Scalar.sqrt(6 / (Tensor.Scalar(input) + Tensor.Scalar(out)))
      let min = -value
      let max = value
      
      return Tensor.Scalar.random(in: min...max)
      
    case .xavierNormal:
      return Tensor.Scalar(dist.gaussRand) * Tensor.Scalar.sqrt(2 / (Tensor.Scalar(input) + Tensor.Scalar(out)))
      
    case .heUniform:
      let value = Tensor.Scalar.sqrt(6 / Tensor.Scalar(input))
      let min = -value
      let max = value
    
      return Tensor.Scalar.random(in: min...max)
      
    case .heNormal:
      return Tensor.Scalar(dist.gaussRand) * Tensor.Scalar.sqrt(2 / (Tensor.Scalar(input)))
      
    case .normal:
      return Tensor.Scalar(dist.gaussRand)
    }
  }
  
  /// Generates a tensor filled with initialized scalar values.
  ///
  /// - Parameters:
  ///   - size: Target tensor shape.
  ///   - input: Fan-in size.
  ///   - out: Fan-out size (used by Xavier variants).
  /// - Returns: Tensor populated with initialized values.
  public func calculate(size: TensorSize, input: Int, out: Int = 0) -> Tensor {
    var tensor: [[[Tensor.Scalar]]] = []
    
    for _ in 0..<size.depth {
      var rows: [[Tensor.Scalar]] = []
      
      for _ in 0..<size.rows {
        var columns: [Tensor.Scalar] = []
        
        for _ in 0..<size.columns {
          columns.append(calculate(input: input, out: out))
        }
        rows.append(columns)
      }
      
      tensor.append(rows)
    }
    
    return Tensor(tensor)
  }
}

extension Initializer: Codable {
  /// Encodes the initializer strategy.
  ///
  /// - Parameter encoder: Encoder used for serialization.
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: Self.CodingKeys.self)
    
    switch self.type {
    case .xavierUniform:
      try container.encode(CodingKeys.xavierUniform.stringValue, forKey: .xavierUniform)
    case .xavierNormal:
      try container.encode(CodingKeys.xavierNormal.stringValue, forKey: .xavierNormal)
    case .heUniform:
      try container.encode(CodingKeys.heUniform.stringValue, forKey: .heUniform)
    case .heNormal:
      try container.encode(CodingKeys.heNormal.stringValue, forKey: .heNormal)
    case .normal(let std):
      try container.encode(CodingKeys.normal.stringValue + "-\(std)", forKey: .normal)
    }
  }
  
  /// Decodes an initializer strategy.
  ///
  /// - Parameter decoder: Decoder containing encoded initializer configuration.
  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    self = .init(type: .heUniform)

    try values.allKeys.forEach { k in
      if k == .normal {
        if let val = try values.decodeIfPresent(String.self, forKey: .normal) {
          let split = val.split(separator: "-")[safe: 1, ""]
          if let std = Tensor.Scalar(split) {
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

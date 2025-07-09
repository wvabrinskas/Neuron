//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/28/22.
//

import Foundation
import GameplayKit

/// Weight initializer methods for neural network layers
/// Different initialization strategies can significantly impact training performance
public enum InitializerType: Codable, Equatable {
  /// Xavier/Glorot normal initialization - generates weights from normal distribution
  /// Good for sigmoid and tanh activations. Mean = 0, variance = 2/(fan_in + fan_out)
  case xavierNormal
  /// Xavier/Glorot uniform initialization - generates weights from uniform distribution
  /// Good for sigmoid and tanh activations. Range = ±√(6/(fan_in + fan_out))
  case xavierUniform
  /// He normal initialization - generates weights from normal distribution
  /// Good for ReLU activations. Mean = 0, variance = 2/fan_in
  case heNormal
  /// He uniform initialization - generates weights from uniform distribution
  /// Good for ReLU activations. Range = ±√(6/fan_in)
  case heUniform
  /// Normal initialization with custom standard deviation
  /// - Parameter std: Standard deviation for the normal distribution
  case normal(std: Tensor.Scalar)
  
  /// Creates an Initializer instance from this type
  /// - Returns: A configured Initializer object
  public func build() -> Initializer {
    Initializer(type: self)
  }
}

/// Weight initializer that generates initial values for neural network parameters
/// Implements various initialization strategies to improve training performance
public struct Initializer {
  /// The initialization type/strategy being used
  public let type: InitializerType
  /// Gaussian distribution generator for normal initialization methods
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
  
  /// Initializes the weight initializer with a specific type
  /// - Parameter type: The initialization strategy to use
  public init(type: InitializerType) {
    self.type = type
    switch type {
    case .normal(let std):
      self.dist = Gaussian(std: Double(std), mean: 0)
    default:
      break
    }
  }
  
  /// Calculates a single weight value based on the initialization strategy
  /// - Parameters:
  ///   - input: Number of input connections (fan-in)
  ///   - out: Number of output connections (fan-out), defaults to 0
  /// - Returns: A single initialized weight value
  public func calculate(input: Int, out: Int = 0) -> Tensor.Scalar {
    switch type {
      
    case .xavierUniform:
      let min = -Tensor.Scalar(sqrt(6) / sqrt((Double(input) + Double(out))))
      let max = Tensor.Scalar(sqrt(6) / sqrt((Double(input) + Double(out))))
      
      return Tensor.Scalar.random(in: min...max)
      
    case .xavierNormal:
      return Tensor.Scalar(dist.gaussRand) * Tensor.Scalar(sqrt(2 / (Double(input) + Double(out))))
      
    case .heUniform:
      let min = -Tensor.Scalar(sqrt(6) / sqrt((Double(input))))
      let max = Tensor.Scalar(sqrt(6) / sqrt((Double(input))))
      
      return Tensor.Scalar.random(in: min...max)
      
    case .heNormal:
      return Tensor.Scalar(dist.gaussRand) * Tensor.Scalar(sqrt(2 / (Double(input))))
      
    case .normal:
      return Tensor.Scalar(dist.gaussRand)
    }
  }
  
  /// Creates a tensor filled with initialized weight values
  /// - Parameters:
  ///   - size: The desired tensor dimensions
  ///   - input: Number of input connections (fan-in)
  ///   - out: Number of output connections (fan-out), defaults to 0
  /// - Returns: A tensor filled with initialized weight values
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

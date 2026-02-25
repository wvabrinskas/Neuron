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
  case orthogonal(gain: Tensor.Scalar)
  case normal(std: Tensor.Scalar)
  
  /// Creates an executable initializer from this initializer type.
  ///
  /// - Returns: `Initializer` wrapper configured for this case.
  public func build() -> Initializer {
    Initializer(type: self)
  }
}

/// A structure that represents a weight initializer, encapsulating an initialization strategy
/// and an optional Gaussian distribution for generating initial weight values.
public struct Initializer {
/// The type of initialization strategy used by this initializer.
  public let type: InitializerType
  private var dist: Gaussian = Gaussian(std: 1, mean: 0)
  private var normal = NormalDistribution(mean: 0, deviation: 1)
  
/// Coding keys used for encoding and decoding the initializer type,
  /// each mapping to a corresponding `InitializerType` value.
  public enum CodingKeys: String, CodingKey, CaseIterable {
    case xavierNormal
    case xavierUniform
    case heNormal
    case heUniform
    case normal
    case orthogonal
    
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
      case .orthogonal:
        return .orthogonal(gain: 1)
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


  /// Generates a tensor filled with initialized scalar values.
  ///
  /// - Parameters:
  ///   - size: Target tensor shape.
  ///   - input: Fan-in size.
  ///   - out: Fan-out size (used by Xavier variants).
  /// - Returns: Tensor populated with initialized values.
  public func calculate(size: TensorSize, input: Int, out: Int = 0) -> Tensor {
    if case .orthogonal(let gain) = type {
      return orthogonalTensor(size: size, gain: gain)
    }

    var tensor: Tensor.Value = .init(repeating: 0, count: size.depth * size.rows * size.columns)

    for d in 0..<size.depth {
      for r in 0..<size.rows {
        for c in 0..<size.columns {
          tensor[d * size.rows * size.columns + r * size.columns + c] = calculate(input: input, out: out)
        }
      }
    }

    return Tensor(tensor, size: size)
  }
  
  /// Generates one initialized scalar value.
  ///
  /// - Parameters:
  ///   - input: Fan-in size.
  ///   - out: Fan-out size (used by Xavier variants).
  /// - Returns: One initialized scalar sampled using the selected strategy.
  private func calculate(input: Int, out: Int = 0) -> Tensor.Scalar {
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
      
    case .orthogonal(let gain):
      // Orthogonal initialization is inherently a matrix-level operation.
      // Per-scalar calls fall back to scaled normal samples; use calculate(size:input:out:) for correct behavior.
      return gain * normal.nextScalar()
    }
  }

  /// Generates an orthogonal weight tensor using Gram-Schmidt QR decomposition.
  ///
  /// Produces a tensor where each depth slice is an orthogonal matrix (or its transpose
  /// when rows < columns), scaled by `gain`. This follows the same approach as
  /// PyTorch's `torch.nn.init.orthogonal_`.
  ///
  /// - Parameters:
  ///   - size: Target tensor shape.
  ///   - gain: Scaling factor applied to the orthogonal matrix.
  /// - Returns: Tensor with orthogonal weight values.
  private func orthogonalTensor(size: TensorSize, gain: Tensor.Scalar) -> Tensor {
    var tensor: [[[Tensor.Scalar]]] = []

    for _ in 0..<size.depth {
      let rows = size.rows
      let cols = size.columns

      // Draw a random matrix from the standard normal distribution.
      let A: [[Tensor.Scalar]] = (0..<rows).map { _ in
        (0..<cols).map { _ in normal.nextScalar() }
      }

      // Gram-Schmidt orthonormalization on the rows of A.
      var Q: [[Tensor.Scalar]] = []
      for i in 0..<rows {
        var v = A[i]
        for q in Q {
          let proj = dot(v, q)
          v = subtract(v, scale(q, by: proj))
        }
        let norm = magnitude(v)
        if norm > 1e-10 {
          Q.append(scale(v, by: 1 / norm))
        } else {
          Q.append([Tensor.Scalar](repeating: 0, count: cols))
        }
      }

      // Scale by gain.
      let result = Q.map { row in row.map { $0 * gain } }
      tensor.append(result)
    }

    return Tensor(tensor)
  }

  private func dot(_ a: [Tensor.Scalar], _ b: [Tensor.Scalar]) -> Tensor.Scalar {
    zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
  }

  private func subtract(_ a: [Tensor.Scalar], _ b: [Tensor.Scalar]) -> [Tensor.Scalar] {
    zip(a, b).map { $0 - $1 }
  }

  private func scale(_ a: [Tensor.Scalar], by s: Tensor.Scalar) -> [Tensor.Scalar] {
    a.map { $0 * s }
  }

  private func magnitude(_ a: [Tensor.Scalar]) -> Tensor.Scalar {
    Tensor.Scalar.sqrt(a.reduce(0) { $0 + $1 * $1 })
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
    case .orthogonal(let gain):
      try container.encode(CodingKeys.orthogonal.stringValue + "-\(gain)", forKey: .orthogonal)
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

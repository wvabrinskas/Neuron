//
//  TensorStorage+Arithmetic.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/4/26.
//

import Foundation
import NumSwift

// MARK: - Element-wise Arithmetic (TensorStorage x TensorStorage)

public extension TensorStorage {

  static func + (lhs: TensorStorage, rhs: TensorStorage) -> TensorStorage {
    let count = Swift.min(lhs.count, rhs.count)
    let result = TensorStorage.create(count: count)
    NumSwiftFlat.add(lhs.pointer, rhs.pointer, result: result.pointer, count: count)
    return result
  }

  static func - (lhs: TensorStorage, rhs: TensorStorage) -> TensorStorage {
    let count = Swift.min(lhs.count, rhs.count)
    let result = TensorStorage.create(count: count)
    NumSwiftFlat.sub(lhs.pointer, rhs.pointer, result: result.pointer, count: count)
    return result
  }

  static func * (lhs: TensorStorage, rhs: TensorStorage) -> TensorStorage {
    let count = Swift.min(lhs.count, rhs.count)
    let result = TensorStorage.create(count: count)
    NumSwiftFlat.mul(lhs.pointer, rhs.pointer, result: result.pointer, count: count)
    return result
  }

  static func / (lhs: TensorStorage, rhs: TensorStorage) -> TensorStorage {
    let count = Swift.min(lhs.count, rhs.count)
    let result = TensorStorage.create(count: count)
    NumSwiftFlat.div(lhs.pointer, rhs.pointer, result: result.pointer, count: count)
    return result
  }
}

// MARK: - Tensor.Scalar Arithmetic (TensorStorage x Tensor.Scalar)

public extension TensorStorage {

  static func + (lhs: TensorStorage, rhs: Tensor.Scalar) -> TensorStorage {
    let result = TensorStorage.create(count: lhs.count)
    NumSwiftFlat.add(lhs.pointer, scalar: rhs, result: result.pointer, count: lhs.count)
    return result
  }

  static func - (lhs: TensorStorage, rhs: Tensor.Scalar) -> TensorStorage {
    let result = TensorStorage.create(count: lhs.count)
    NumSwiftFlat.sub(lhs.pointer, scalar: rhs, result: result.pointer, count: lhs.count)
    return result
  }

  static func * (lhs: TensorStorage, rhs: Tensor.Scalar) -> TensorStorage {
    let result = TensorStorage.create(count: lhs.count)
    NumSwiftFlat.mul(lhs.pointer, scalar: rhs, result: result.pointer, count: lhs.count)
    return result
  }

  static func / (lhs: TensorStorage, rhs: Tensor.Scalar) -> TensorStorage {
    let result = TensorStorage.create(count: lhs.count)
    NumSwiftFlat.div(lhs.pointer, scalar: rhs, result: result.pointer, count: lhs.count)
    return result
  }
}

// MARK: - Tensor.Scalar Arithmetic (Tensor.Scalar x TensorStorage)

public extension TensorStorage {

  static func * (lhs: Tensor.Scalar, rhs: TensorStorage) -> TensorStorage {
    let result = TensorStorage.create(count: rhs.count)
    NumSwiftFlat.mul(rhs.pointer, scalar: lhs, result: result.pointer, count: rhs.count)
    return result
  }

  static func - (lhs: Tensor.Scalar, rhs: TensorStorage) -> TensorStorage {
    let result = TensorStorage.create(count: rhs.count)
    NumSwiftFlat.sub(scalar: lhs, rhs.pointer, result: result.pointer, count: rhs.count)
    return result
  }

  static func / (lhs: Tensor.Scalar, rhs: TensorStorage) -> TensorStorage {
    let result = TensorStorage.create(count: rhs.count)
    NumSwiftFlat.div(scalar: lhs, rhs.pointer, result: result.pointer, count: rhs.count)
    return result
  }

  static func + (lhs: Tensor.Scalar, rhs: TensorStorage) -> TensorStorage {
    let result = TensorStorage.create(count: rhs.count)
    NumSwiftFlat.add(rhs.pointer, scalar: lhs, result: result.pointer, count: rhs.count)
    return result
  }
}

// MARK: - Reductions

public extension TensorStorage {

  /// Sum of all elements.
  var sum: Tensor.Scalar {
    guard count > 0 else { return 0 }
    return NumSwiftFlat.sum(pointer, count: count)
  }

  /// Mean of all elements.
  var mean: Tensor.Scalar {
    guard count > 0 else { return 0 }
    return NumSwiftFlat.mean(pointer, count: count)
  }

  /// Sum of squares of all elements.
  var sumOfSquares: Tensor.Scalar {
    guard count > 0 else { return 0 }
    return NumSwiftFlat.sumOfSquares(pointer, count: count)
  }
}

// MARK: - Unary Operations

public extension TensorStorage {

  /// Returns a new TensorStorage with every element negated.
  func negated() -> TensorStorage {
    let result = TensorStorage.create(count: count)
    guard count > 0 else { return result }
    NumSwiftFlat.negate(pointer, result: result.pointer, count: count)
    return result
  }

  /// Returns a new TensorStorage with element-wise square root.
  func squareRoot() -> TensorStorage {
    let result = TensorStorage.create(count: count)
    guard count > 0 else { return result }
    NumSwiftFlat.sqrt(pointer, result: result.pointer, count: count)
    return result
  }

  /// Returns a new TensorStorage with elements clamped to `[-limit, limit]`.
  func clipped(to limit: Tensor.Scalar) -> TensorStorage {
    let result = TensorStorage.create(count: count)
    guard count > 0 else { return result }
    NumSwiftFlat.clip(pointer, result: result.pointer, count: count, limit: limit)
    return result
  }
}

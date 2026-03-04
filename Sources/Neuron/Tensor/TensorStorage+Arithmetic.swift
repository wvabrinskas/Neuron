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
    precondition(lhs.count == rhs.count, "TensorStorage count mismatch: \(lhs.count) vs \(rhs.count)")
    let result = TensorStorage(count: lhs.count)
    NumSwiftFlat.add(lhs.pointer, rhs.pointer, result: result.pointer, count: lhs.count)
    return result
  }

  static func - (lhs: TensorStorage, rhs: TensorStorage) -> TensorStorage {
    precondition(lhs.count == rhs.count, "TensorStorage count mismatch: \(lhs.count) vs \(rhs.count)")
    let result = TensorStorage(count: lhs.count)
    NumSwiftFlat.sub(lhs.pointer, rhs.pointer, result: result.pointer, count: lhs.count)
    return result
  }

  static func * (lhs: TensorStorage, rhs: TensorStorage) -> TensorStorage {
    precondition(lhs.count == rhs.count, "TensorStorage count mismatch: \(lhs.count) vs \(rhs.count)")
    let result = TensorStorage(count: lhs.count)
    NumSwiftFlat.mul(lhs.pointer, rhs.pointer, result: result.pointer, count: lhs.count)
    return result
  }

  static func / (lhs: TensorStorage, rhs: TensorStorage) -> TensorStorage {
    precondition(lhs.count == rhs.count, "TensorStorage count mismatch: \(lhs.count) vs \(rhs.count)")
    let result = TensorStorage(count: lhs.count)
    NumSwiftFlat.div(lhs.pointer, rhs.pointer, result: result.pointer, count: lhs.count)
    return result
  }
}

// MARK: - Scalar Arithmetic (TensorStorage x Scalar)

public extension TensorStorage {

  static func + (lhs: TensorStorage, rhs: Scalar) -> TensorStorage {
    let result = TensorStorage(count: lhs.count)
    NumSwiftFlat.add(lhs.pointer, scalar: rhs, result: result.pointer, count: lhs.count)
    return result
  }

  static func - (lhs: TensorStorage, rhs: Scalar) -> TensorStorage {
    let result = TensorStorage(count: lhs.count)
    NumSwiftFlat.sub(lhs.pointer, scalar: rhs, result: result.pointer, count: lhs.count)
    return result
  }

  static func * (lhs: TensorStorage, rhs: Scalar) -> TensorStorage {
    let result = TensorStorage(count: lhs.count)
    NumSwiftFlat.mul(lhs.pointer, scalar: rhs, result: result.pointer, count: lhs.count)
    return result
  }

  static func / (lhs: TensorStorage, rhs: Scalar) -> TensorStorage {
    let result = TensorStorage(count: lhs.count)
    NumSwiftFlat.div(lhs.pointer, scalar: rhs, result: result.pointer, count: lhs.count)
    return result
  }
}

// MARK: - Scalar Arithmetic (Scalar x TensorStorage)

public extension TensorStorage {

  static func * (lhs: Scalar, rhs: TensorStorage) -> TensorStorage {
    let result = TensorStorage(count: rhs.count)
    NumSwiftFlat.mul(rhs.pointer, scalar: lhs, result: result.pointer, count: rhs.count)
    return result
  }

  static func - (lhs: Scalar, rhs: TensorStorage) -> TensorStorage {
    let result = TensorStorage(count: rhs.count)
    NumSwiftFlat.sub(scalar: lhs, rhs.pointer, result: result.pointer, count: rhs.count)
    return result
  }

  static func / (lhs: Scalar, rhs: TensorStorage) -> TensorStorage {
    let result = TensorStorage(count: rhs.count)
    NumSwiftFlat.div(scalar: lhs, rhs.pointer, result: result.pointer, count: rhs.count)
    return result
  }

  static func + (lhs: Scalar, rhs: TensorStorage) -> TensorStorage {
    let result = TensorStorage(count: rhs.count)
    NumSwiftFlat.add(rhs.pointer, scalar: lhs, result: result.pointer, count: rhs.count)
    return result
  }
}

// MARK: - Reductions

public extension TensorStorage {

  /// Sum of all elements.
  var sum: Scalar {
    guard count > 0 else { return 0 }
    return NumSwiftFlat.sum(pointer, count: count)
  }

  /// Mean of all elements.
  var mean: Scalar {
    guard count > 0 else { return 0 }
    return NumSwiftFlat.mean(pointer, count: count)
  }

  /// Sum of squares of all elements.
  var sumOfSquares: Scalar {
    guard count > 0 else { return 0 }
    return NumSwiftFlat.sumOfSquares(pointer, count: count)
  }
}

// MARK: - Unary Operations

public extension TensorStorage {

  /// Returns a new TensorStorage with every element negated.
  func negated() -> TensorStorage {
    let result = TensorStorage(count: count)
    guard count > 0 else { return result }
    NumSwiftFlat.negate(pointer, result: result.pointer, count: count)
    return result
  }

  /// Returns a new TensorStorage with element-wise square root.
  func squareRoot() -> TensorStorage {
    let result = TensorStorage(count: count)
    guard count > 0 else { return result }
    NumSwiftFlat.sqrt(pointer, result: result.pointer, count: count)
    return result
  }

  /// Returns a new TensorStorage with elements clamped to `[-limit, limit]`.
  func clipped(to limit: Scalar) -> TensorStorage {
    let result = TensorStorage(count: count)
    guard count > 0 else { return result }
    NumSwiftFlat.clip(pointer, result: result.pointer, count: count, limit: limit)
    return result
  }
}

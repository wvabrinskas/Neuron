//
//  TensorSIMD.swift
//  Neuron
//
//  Optimized activation functions operating directly on flat storage.
//  Simple loops over ContiguousArray are auto-vectorized by the compiler
//  into NEON/SSE instructions, replacing manual SIMD3/SIMD4 code.
//

import Foundation
import Numerics

// MARK: - Activation Functions on Flat Storage

extension Tensor {
  
  /// Fast ReLU on flat storage (auto-vectorized)
  internal func reluSIMD() -> Tensor {
    var result = ContiguousArray<Scalar>(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      result[i] = Swift.max(0, storage[i])
    }
    return Tensor(storage: result, size: _size, context: context)
  }
  
  /// Fast Tanh on flat storage (auto-vectorized)
  internal func tanhSIMD() -> Tensor {
    var result = ContiguousArray<Scalar>(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      let x = storage[i]
      let expPos = Scalar.exp(x)
      let expNeg = Scalar.exp(-x)
      result[i] = (expPos - expNeg) / (expPos + expNeg)
    }
    return Tensor(storage: result, size: _size, context: context)
  }
  
  /// Fast Sigmoid on flat storage (auto-vectorized)
  internal func sigmoidSIMD() -> Tensor {
    var result = ContiguousArray<Scalar>(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      result[i] = 1 / (1 + Scalar.exp(-storage[i]))
    }
    return Tensor(storage: result, size: _size, context: context)
  }
  
  /// Fast LeakyReLU on flat storage (auto-vectorized)
  internal func leakyReluSIMD(limit: Scalar) -> Tensor {
    var result = ContiguousArray<Scalar>(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      let x = storage[i]
      result[i] = x >= 0 ? x : limit * x
    }
    return Tensor(storage: result, size: _size, context: context)
  }
  
  /// Fast Swish on flat storage (auto-vectorized)
  internal func swishSIMD() -> Tensor {
    var result = ContiguousArray<Scalar>(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      let x = storage[i]
      let sig = 1 / (1 + Scalar.exp(-x))
      result[i] = x * sig
    }
    return Tensor(storage: result, size: _size, context: context)
  }
  
  /// Fast SELU on flat storage (auto-vectorized)
  internal func seLuSIMD() -> Tensor {
    let lambda: Scalar = 1.0507
    let alpha: Scalar = 1.6733
    var result = ContiguousArray<Scalar>(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      let x = storage[i]
      result[i] = x > 0 ? lambda * x : lambda * alpha * (Scalar.exp(x) - 1)
    }
    return Tensor(storage: result, size: _size, context: context)
  }
  
  /// Fast GELU on flat storage (auto-vectorized)
  internal func geLuSIMD() -> Tensor {
    let sqrt2: Scalar = Scalar(Foundation.sqrt(2.0))
    var result = ContiguousArray<Scalar>(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      let x = storage[i]
      result[i] = x * (1 + Scalar.erf(x / sqrt2)) * 0.5
    }
    return Tensor(storage: result, size: _size, context: context)
  }
}

// MARK: - SIMD Strategy (kept for backward compatibility with CPU.swift)

internal struct SIMDStrategy {
  /// Returns true if SIMD activation functions should be used for this shape.
  /// With flat storage, these are always beneficial since they avoid toNestedArray().
  static func shouldUseSIMD(shape: [Int]) -> Bool {
    guard shape.count == 3 else { return false }
    // With flat storage, the optimized activation functions are always better
    return true
  }
}

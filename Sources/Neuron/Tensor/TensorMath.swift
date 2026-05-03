//
//  File.swift
//
//
//  Created by William Vabrinskas on 6/27/22.
//

import Foundation
import NumSwift

public extension Float {
  /// A small constant added to denominators and square-roots for numerical stability.
  ///
  /// Value is `1e-12`. Use `Tensor.Scalar.stabilityFactor` in generic contexts so
  /// that code works for both `Float` and `Float16`.
  static var stabilityFactor: Self {
    1e-12
  }
}

#if arch(arm64)
public extension Float16 {
  /// A small constant added to denominators and square-roots for numerical stability.
  ///
  /// Uses a larger value (`1e-4`) than the `Float` equivalent to account for the
  /// reduced dynamic range of `Float16`. Only available on arm64.
  static var stabilityFactor: Self {
    1e-4
  }
}
#endif

public extension Tensor {
  /// A closure type that receives a pointer to a contiguous block of scalars and its count,
  /// and returns a single reduced scalar result (e.g., sum, mean, or max).
  ///
  /// Used by `apply(axis:_:)` to perform axis-wise reductions over the tensor.
  typealias PointerMathBlock = (_ ptr: TensorStorage.Pointer, _ count: Int) -> Scalar
  
  /// Applies a reduction closure along the specified axis, returning a tensor with that
  /// dimension collapsed to size 1.
  ///
  /// Axis semantics for a tensor of shape `(columns A, rows B, depth C)`:
  /// - `0`: reduce along rows (Y axis) → shape `(A, 1, C)`
  /// - `1`: reduce along columns (X axis) → shape `(1, B, C)`
  /// - `2`: reduce along depth (Z axis) → shape `(A, B, 1)`
  /// - `-1`: reduce all elements → scalar shape `(1, 1, 1)`
  ///
  /// - Parameters:
  ///   - axis: The dimension to reduce along (`0`, `1`, `2`, or `-1` for global).
  ///   - block: A `PointerMathBlock` closure that maps a contiguous pointer and element
  ///            count to a single reduced `Tensor.Scalar`.
  /// - Returns: A new `Tensor` with the specified dimension collapsed.
  func apply(axis: Int, _ block: PointerMathBlock) -> Tensor {
    let columns = size.columns
    let rows = size.rows
    let depth = size.depth
    let selfPtr = storage.pointer
    
    if axis == 0 {
      // Reduce along rows -> output (columns x 1 x depth)
      let outSize = TensorSize(rows: 1, columns: columns, depth: depth)
      let outStorage = TensorStorage.create(count: columns * depth)
      let outPtr = outStorage.pointer
      let scratch = TensorStorage.create(count: rows)
      let scratchPtr = scratch.pointer
      
      for d in 0..<depth {
        for c in 0..<columns {
          for r in 0..<rows {
            scratchPtr[r] = selfPtr[flatIndex(column: c, row: r, depth: d)]
          }
          outPtr[d * columns + c] = block(scratchPtr, rows)
        }
      }
      
      return Tensor(storage: outStorage, size: outSize)
      
    } else if axis == 1 {
      // Reduce along columns -> output (1 x rows x depth)
      let outSize = TensorSize(rows: rows, columns: 1, depth: depth)
      let outStorage = TensorStorage.create(count: rows * depth)
      let outPtr = outStorage.pointer
      
      for d in 0..<depth {
        for r in 0..<rows {
          let start = flatIndex(column: 0, row: r, depth: d)
          outPtr[d * rows + r] = block(selfPtr + start, columns)
        }
      }
      
      return Tensor(storage: outStorage, size: outSize)
      
    } else if axis == 2 {
      // Reduce along depth -> output (columns x rows x 1)
      let outSize = TensorSize(rows: rows, columns: columns, depth: 1)
      let outStorage = TensorStorage.create(count: columns * rows)
      let outPtr = outStorage.pointer
      let scratch = TensorStorage.create(count: depth)
      let scratchPtr = scratch.pointer
      
      for r in 0..<rows {
        for c in 0..<columns {
          for d in 0..<depth {
            scratchPtr[d] = selfPtr[flatIndex(column: c, row: r, depth: d)]
          }
          outPtr[r * columns + c] = block(scratchPtr, depth)
        }
      }
      
      return Tensor(storage: outStorage, size: outSize)
    }
    
    return Tensor(storage: storage, size: size)
  }
  
  /// Determines the appropriate axis for broadcasting operations between two tensors.
  /// This helper function is used by the arithmetic operators (+, -, *, /) to determine
  /// if broadcasting is possible and which axis should be used.
  ///
  /// - Parameters:
  ///   - selfSize: The size of the first tensor
  ///   - size: The size of the second tensor
  /// - Returns: The axis along which broadcasting should be applied, or nil if broadcasting is not possible
  ///
  /// Broadcasting rules:
  /// - Returns 0 if broadcasting along rows (Y axis)
  /// - Returns 1 if broadcasting along columns (X axis)
  /// - Returns 2 if broadcasting along depth (Z axis)
  /// - Returns nil if tensors are not compatible for broadcasting
  static func axisToApplyAlong(selfSize: TensorSize, size: TensorSize) -> Int? {
    if size.columns == selfSize.columns,
       size.rows == 1,
       size.depth == selfSize.depth {
      return 0
      
    } else if size.columns == 1,
              size.rows == selfSize.rows,
              size.depth == selfSize.depth {
      return 1
      
    } else if size.columns == selfSize.columns,
              size.rows == selfSize.rows,
              size.depth == 1 {
      return 2
    } else {
      return nil
    }
  }

  private enum BroadcastOp { case add, sub, mul, div }

  /// Per-channel broadcasting fast path for (W,H,D) op (1,1,D).
  /// Applies a per-depth scalar from `value` across each spatial slice of `self`.
  private func broadcastPerChannelFastPath(value: Tensor, op: BroadcastOp) -> Tensor? {
    let selfSize = size
    let inputSize = value.size
    guard inputSize.columns == 1,
          inputSize.rows == 1,
          inputSize.depth == selfSize.depth else { return nil }

    let columns = selfSize.columns
    let rows = selfSize.rows
    let depth = selfSize.depth
    let sliceSize = columns * rows
    let totalCount = sliceSize * depth
    guard totalCount > 0 else { return nil }

    let resultStorage = TensorStorage.create(count: totalCount)
    let selfPtr = storage.pointer
    let inputPtr = value.storage.pointer
    let resultPtr = resultStorage.pointer

    for d in 0..<depth {
      let offset = d * sliceSize
      let scalar = inputPtr[d]
      switch op {
      case .add: NumSwiftFlat.add(selfPtr + offset, scalar: scalar, result: resultPtr + offset, count: sliceSize)
      case .sub: NumSwiftFlat.sub(selfPtr + offset, scalar: scalar, result: resultPtr + offset, count: sliceSize)
      case .mul: NumSwiftFlat.mul(selfPtr + offset, scalar: scalar, result: resultPtr + offset, count: sliceSize)
      case .div: NumSwiftFlat.div(selfPtr + offset, scalar: scalar, result: resultPtr + offset, count: sliceSize)
      }
    }

    let context = perChannelBroadcastContext(value: value, op: op)
    return Tensor(storage: resultStorage, size: selfSize, context: context)
  }

  /// Specialized context for (H,W,C) op (1,1,C) broadcast operations.
  /// Correctly reduces gradients spatially (sum over H,W) when backpropagating
  /// through the `value (1,1,C)` path, avoiding the shape mismatch that the
  /// generic arithmetic contexts produce.
  private func perChannelBroadcastContext(value: Tensor, op: BroadcastOp) -> TensorContext {
    // `self` is (H,W,C), `value` is (1,1,C)
    let selfCapture = self
    let valueSize = value.size

    func reduceSpatial(_ t: Tensor) -> Tensor {
      let depth = valueSize.depth
      let sliceSize = t.size.rows * t.size.columns
      let sumStorage = TensorStorage.create(count: depth)
      let srcPtr = t.storage.pointer
      for d in 0..<depth {
        var s: Tensor.Scalar = 0
        let base = d * sliceSize
        for i in 0..<sliceSize { s += srcPtr[base + i] }
        sumStorage[d] = s
      }
      return Tensor(storage: sumStorage, size: valueSize)
    }

    return TensorContext { inputs, gradient, wrt in
      if value.graphChain.contains(wrt.id) || value.id == wrt.id {
        // Gradient w.r.t. value (1,1,C) — reduce spatial dims back to (1,1,C)
        switch op {
        case .add:
          return (reduceSpatial(gradient), Tensor(), Tensor())
        case .sub:
          return (reduceSpatial(gradient * Tensor.Scalar(-1)), Tensor(), Tensor())
        case .mul:
          return (reduceSpatial(gradient * selfCapture.copy()), Tensor(), Tensor())
        case .div:
          let valueSq = value * value
          let ratio = selfCapture.copy() / valueSq
          return (reduceSpatial(gradient * ratio * Tensor.Scalar(-1)), Tensor(), Tensor())
        }
      } else {
        // Gradient w.r.t. self (H,W,C) — broadcast, no reduction needed
        switch op {
        case .add:
          return (gradient, Tensor(), Tensor())
        case .sub:
          return (gradient, Tensor(), Tensor())
        case .mul:
          return (gradient * value.copy(), Tensor(), Tensor())
        case .div:
          return (gradient / value.copy(), Tensor(), Tensor())
        }
      }
    }
  }

  /// Pointer-based fast path for *Along broadcasting. Returns result storage when applicable, nil to fall back.
  private func broadcastAlongFastPath(axis: Int, value: Tensor, op: BroadcastOp) -> Tensor? {
    let inputSize = value.size
    let selfSize = size
    let columns = selfSize.columns
    let rows = selfSize.rows
    let depth = selfSize.depth
    let totalCount = columns * rows * depth
    guard totalCount > 0 else { return nil }

    let resultStorage = TensorStorage.create(count: totalCount)
    let selfPtr = storage.pointer
    let inputPtr = value.storage.pointer
    let resultPtr = resultStorage.pointer

    switch op {
    case .add:
      if axis == 0, inputSize.columns == columns, inputSize.rows == 1, inputSize.depth == depth {
        for d in 0..<depth {
          let inputStart = value.flatIndex(column: 0, row: 0, depth: d)
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            NumSwiftFlat.add(selfPtr + selfStart, inputPtr + inputStart, result: resultPtr + selfStart, count: columns)
          }
        }
      } else if axis == 1, inputSize.columns == 1, inputSize.rows == rows, inputSize.depth == depth {
        for d in 0..<depth {
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            let inputIdx = value.flatIndex(column: 0, row: r, depth: d)
            NumSwiftFlat.add(selfPtr + selfStart, scalar: inputPtr[inputIdx], result: resultPtr + selfStart, count: columns)
          }
        }
      } else if axis == 2, inputSize.columns == columns, inputSize.rows == rows, inputSize.depth == 1 {
        for d in 0..<depth {
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            let inputStart = value.flatIndex(column: 0, row: r, depth: 0)
            NumSwiftFlat.add(selfPtr + selfStart, inputPtr + inputStart, result: resultPtr + selfStart, count: columns)
          }
        }
      } else { return nil }
    case .sub:
      if axis == 0, inputSize.columns == columns, inputSize.rows == 1, inputSize.depth == depth {
        for d in 0..<depth {
          let inputStart = value.flatIndex(column: 0, row: 0, depth: d)
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            NumSwiftFlat.sub(selfPtr + selfStart, inputPtr + inputStart, result: resultPtr + selfStart, count: columns)
          }
        }
      } else if axis == 1, inputSize.columns == 1, inputSize.rows == rows, inputSize.depth == depth {
        for d in 0..<depth {
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            let inputIdx = value.flatIndex(column: 0, row: r, depth: d)
            NumSwiftFlat.sub(selfPtr + selfStart, scalar: inputPtr[inputIdx], result: resultPtr + selfStart, count: columns)
          }
        }
      } else if axis == 2, inputSize.columns == columns, inputSize.rows == rows, inputSize.depth == 1 {
        for d in 0..<depth {
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            let inputStart = value.flatIndex(column: 0, row: r, depth: 0)
            NumSwiftFlat.sub(selfPtr + selfStart, inputPtr + inputStart, result: resultPtr + selfStart, count: columns)
          }
        }
      } else { return nil }
    case .mul:
      if axis == 0, inputSize.columns == columns, inputSize.rows == 1, inputSize.depth == depth {
        for d in 0..<depth {
          let inputStart = value.flatIndex(column: 0, row: 0, depth: d)
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            NumSwiftFlat.mul(selfPtr + selfStart, inputPtr + inputStart, result: resultPtr + selfStart, count: columns)
          }
        }
      } else if axis == 1, inputSize.columns == 1, inputSize.rows == rows, inputSize.depth == depth {
        for d in 0..<depth {
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            let inputIdx = value.flatIndex(column: 0, row: r, depth: d)
            NumSwiftFlat.mul(selfPtr + selfStart, scalar: inputPtr[inputIdx], result: resultPtr + selfStart, count: columns)
          }
        }
      } else if axis == 2, inputSize.columns == columns, inputSize.rows == rows, inputSize.depth == 1 {
        for d in 0..<depth {
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            let inputStart = value.flatIndex(column: 0, row: r, depth: 0)
            NumSwiftFlat.mul(selfPtr + selfStart, inputPtr + inputStart, result: resultPtr + selfStart, count: columns)
          }
        }
      } else { return nil }
    case .div:
      // NumSwiftFlat.div(a, b, result) gives result = a/b (see NumSwift implementation)
      if axis == 0, inputSize.columns == columns, inputSize.rows == 1, inputSize.depth == depth {
        for d in 0..<depth {
          let inputStart = value.flatIndex(column: 0, row: 0, depth: d)
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            NumSwiftFlat.div(selfPtr + selfStart, inputPtr + inputStart, result: resultPtr + selfStart, count: columns)
          }
        }
      } else if axis == 1, inputSize.columns == 1, inputSize.rows == rows, inputSize.depth == depth {
        for d in 0..<depth {
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            let inputIdx = value.flatIndex(column: 0, row: r, depth: d)
            NumSwiftFlat.div(selfPtr + selfStart, scalar: inputPtr[inputIdx], result: resultPtr + selfStart, count: columns)
          }
        }
      } else if axis == 2, inputSize.columns == columns, inputSize.rows == rows, inputSize.depth == 1 {
        for d in 0..<depth {
          for r in 0..<rows {
            let selfStart = flatIndex(column: 0, row: r, depth: d)
            let inputStart = value.flatIndex(column: 0, row: r, depth: 0)
            NumSwiftFlat.div(selfPtr + selfStart, inputPtr + inputStart, result: resultPtr + selfStart, count: columns)
          }
        }
      } else { return nil }
    }

    let context: TensorContext
    switch op {
    case .add: context = addContext(value: value)
    case .sub: context = subtractContext(value: value)
    case .mul: context = multiplyContext(value: value)
    case .div: context = divideContext(value: value)
    }
    return Tensor(storage: resultStorage, size: selfSize, context: context)
  }
  
  func divideContext(value: Tensor) -> TensorContext {
    let branchNode: Tensor? = if sharesGraph(with: value) {
      if self.graphChain.contains(value.id) {
        value
      } else {
        self
      }
    } else {
      nil
    }
    
    // in this context `self` is the other half of the equation. it's lhs. value is rhs
    return TensorContext { inputs, gradient, wrt in
      if value.graphChain.contains(wrt.id) || value.id == wrt.id {
        let gradB = gradient * (-1 * (self.copy() / (value * value)))
        gradB.label = "division_grad_b"
        branchNode?.setGradientBranch(gradB)
        return (gradB, Tensor(), Tensor())
      } else {
        let gradA = gradient / value
        gradA.label = "division_grad_a"
        branchNode?.setGradientBranch(gradA)
        return (gradA, Tensor(), Tensor())
      }
    }
  }
  
  /// Performs element-wise division along a specific axis with broadcasting support.
  ///
  /// - Parameters:
  ///   - axis: The axis along which to perform the division (0, 1, or 2)
  ///   - value: The tensor to divide by, which will be broadcast along the specified axis
  /// - Returns: A new tensor with the result of the division operation
  ///
  /// - Note: Self-assignment is now handled automatically. The operation detects and prevents
  ///   reference cycles in the computation graph, so manual `.copy()` calls are no longer required.
  func divideAlong(axis: Int, value: Tensor) -> Tensor {
    if let new = broadcastAlongFastPath(axis: axis, value: value, op: .div) {
      new.label = "division"
      if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
      else { new.setGraphSafe(value); new.setGraphSafe(self) }
      return new
    }
    return Tensor(storage: storage.copy(), size: size, context: divideContext(value: value))
  }
  
  func multiplyContext(value: Tensor) -> TensorContext {
    let branchNode: Tensor? = if sharesGraph(with: value) {
      if self.graphChain.contains(value.id) {
        value
      } else {
        self
      }
    } else {
      nil
    }
  
    // in this context `self` is the other half of the equation. it's lhs. value is rhs
    return TensorContext { inputs, gradient, wrt in
      if value.graphChain.contains(wrt.id) || value.id == wrt.id {
        let gradB = gradient * self.copy()
        gradB.label = "multiplication_grad_b"
        branchNode?.setGradientBranch(gradB)
        return (gradB, Tensor(), Tensor())
      } else {
        let gradA = gradient * value.copy()
        gradA.label = "multiplication_grad_a"
        branchNode?.setGradientBranch(gradA)
        return (gradA, Tensor(), Tensor())
      }
    }
  }
  
  /// Performs element-wise multiplication along a specific axis with broadcasting support.
  ///
  /// - Parameters:
  ///   - axis: The axis along which to perform the multiplication (0, 1, or 2)
  ///   - value: The tensor to multiply, which will be broadcast along the specified axis
  /// - Returns: A new tensor with the result of the multiplication operation
  ///
  /// - Note: Self-assignment is now handled automatically. The operation detects and prevents
  ///   reference cycles in the computation graph, so manual `.copy()` calls are no longer required.
  func multiplyAlong(axis: Int, value: Tensor) -> Tensor {
    if let new = broadcastAlongFastPath(axis: axis, value: value, op: .mul) {
      new.label = "multiplication"
      if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
      else { new.setGraphSafe(value); new.setGraphSafe(self) }
      return new
    }
    return Tensor(storage: storage.copy(), size: size, context: multiplyContext(value: value))
  }
  
  func addContext(value: Tensor) -> TensorContext {
    let branchNode: Tensor? = if sharesGraph(with: value) {
      /*
        while this works it doesn't account for long chain branches as well
        we might need to add branch gradient setting in every backprop context??
       this logic will actually fail to apply the right branching logic since value isn't
       a part of self.graphChain so it'll apply gradient branch to self, which will result in
       incorrect gradients.
          1
          |
          2
          | \
          3  1'
          |  |
          4  2'
          \  /
         (4 + 2)
           |
           5
           |
          out
       
        maybe we add another indicator to determine where branch started?
       */
      if self.graphChain.contains(value.id) {
        value
      } else {
        self
      }
    } else {
      nil
    }
    
    return TensorContext { inputs, gradient, wrt in
      let copy = gradient.copy()
      copy.label = "addition_grad"
      branchNode?.setGradientBranch(copy)
      
      return (copy, Tensor(), Tensor())
    }
  }
  
  /// Performs element-wise addition along a specific axis with broadcasting support.
  ///
  /// - Parameters:
  ///   - axis: The axis along which to perform the addition (0, 1, or 2)
  ///   - value: The tensor to add, which will be broadcast along the specified axis
  /// - Returns: A new tensor with the result of the addition operation
  ///
  /// - Note: Self-assignment is now handled automatically. The operation detects and prevents
  ///   reference cycles in the computation graph, so manual `.copy()` calls are no longer required.
  func addAlong(axis: Int, value: Tensor) -> Tensor {
    if let new = broadcastAlongFastPath(axis: axis, value: value, op: .add) {
      new.label = "addition"
      if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
      else { new.setGraphSafe(value); new.setGraphSafe(self) }
      return new
    }
    return Tensor(storage: storage.copy(), size: size, context: addContext(value: value))
  }
  
  func subtractContext(value: Tensor) -> TensorContext {
    let branchNode: Tensor? = if sharesGraph(with: value) {
      if self.graphChain.contains(value.id) {
        value
      } else {
        self
      }
    } else {
      nil
    }
    
    return TensorContext { inputs, gradient, wrt in
      if value.graphChain.contains(wrt.id) || value.id == wrt.id {
        let gradB = gradient * -1
        gradB.label = "subtraction_grad_b"
        
        branchNode?.setGradientBranch(gradB)
        
        return (gradB, Tensor(), Tensor())
      } else {
        let gradA = gradient
        gradA.label = "subtraction_grad_a"
        
        branchNode?.setGradientBranch(gradA)
        return (gradA, Tensor(), Tensor())
      }
    }
  }
  
  /// Performs element-wise subtraction along a specific axis with broadcasting support.
  ///
  /// - Parameters:
  ///   - axis: The axis along which to perform the subtraction (0, 1, or 2)
  ///   - value: The tensor to subtract, which will be broadcast along the specified axis
  /// - Returns: A new tensor with the result of the subtraction operation
  ///
  /// - Note: Self-assignment is now handled automatically. The operation detects and prevents
  ///   reference cycles in the computation graph, so manual `.copy()` calls are no longer required.
  func subtractAlong(axis: Int, value: Tensor) -> Tensor {
    if let new = broadcastAlongFastPath(axis: axis, value: value, op: .sub) {
      new.label = "subtraction"
      if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
      else { new.setGraphSafe(value); new.setGraphSafe(self) }
      return new
    }
    return Tensor(storage: storage.copy(), size: size, context: subtractContext(value: value))
  }
  
  /// Returns the sum of all scalar elements in the tensor.
  /// - Returns: The sum of every element across all dimensions.
  func sum() -> Scalar {
    storage.sum
  }

  /// Asserts (in debug builds) that no element exceeds `limit`.
  ///
  /// - Parameter limit: The maximum allowed scalar value.
  func testLarge(limit: Scalar) {
    for val in storage {
      if val > limit {
        assertionFailure()
        return
      }
    }
  }

  /// Asserts (in debug builds) that all elements are normal floating-point values.
  ///
  /// Triggers an assertion failure and prints the offending value when a
  /// denormalized, infinite, or NaN element is found.
  func testInvalid() {
    for val in storage {
      if val.isNormal == false {
        print(val)
        assertionFailure()
        return
      }
    }
  }

  /// Asserts (in debug builds) that no element is infinite.
  func testInf() {
    for val in storage {
      if val.isInfinite {
        assertionFailure()
        return
      }
    }
  }

  /// Asserts (in debug builds) that no element is NaN.
  func testNaN() {
    for val in storage {
      if val.isNaN {
        assertionFailure()
        return
      }
    }
  }

  /// Computes the batched matrix multiplication `self @ with` for matching depths.
  ///
  /// Requires `self.size.columns == with.size.rows` and `self.size.depth == with.size.depth`.
  ///
  /// - Parameter with: The right-hand matrix tensor.
  /// - Returns: A new tensor of shape `(self.rows, with.columns, depth)`.
  func matmul(_ with: Tensor) -> Tensor {
    let aSize = self.size
    let bSize = with.size
    
    precondition(aSize.columns == bSize.rows, "A columns (\(aSize.columns)) must equal B rows (\(bSize.rows))")
    precondition(aSize.depth == bSize.depth, "A depth (\(aSize.depth)) must equal B depth (\(bSize.depth))")
    
    let depth = aSize.depth
    let aRows = aSize.rows
    let aCols = aSize.columns
    let bCols = bSize.columns
    let bRows = bSize.rows
    let aSliceSize = aRows * aCols
    let bSliceSize = bRows * bCols
    let cSliceSize = aRows * bCols
    
    if depth == 1,
       storage.count == aSliceSize,
       with.storage.count == bSliceSize {
      let resultStorage = TensorStorage.create(count: cSliceSize)
      NumSwiftFlat.matmul(storage.pointer, with.storage.pointer, result: resultStorage.pointer,
                          aRows: aRows, aCols: aCols, bRows: bRows, bCols: bCols)
      let outSize = TensorSize(rows: aRows, columns: bCols, depth: 1)
      return Tensor(storage: resultStorage, size: outSize)
    }
    
    let resultStorage = TensorStorage.create(count: cSliceSize * depth)
    
    for d in 0..<depth {
      let aOffset = d * aSliceSize
      let bOffset = d * bSliceSize
      let cOffset = d * cSliceSize
      
      NumSwiftFlat.matmul(storage.pointer + aOffset, with.storage.pointer + bOffset,
                          result: resultStorage.pointer + cOffset,
                          aRows: aRows, aCols: aCols, bRows: bRows, bCols: bCols)
    }
    
    let outSize = TensorSize(rows: aRows, columns: bCols, depth: depth)
    return Tensor(storage: resultStorage, size: outSize)
  }
  
  /// Returns the sum of squared elements, either globally or along a specific axis.
  ///
  /// - Parameter axis: The axis along which to compute the sum of squares. Pass `-1` (default)
  ///   to reduce all elements to a single scalar tensor.
  /// - Returns: A `Tensor` containing the sum-of-squares result.
  func sumOfSquares(axis: Int = -1) -> Tensor {
    if axis == -1 {
      return Tensor(storage.sumOfSquares)
    }
    
    return apply(axis: axis) { ptr, count in
      NumSwiftFlat.sumOfSquares(ptr, count: count)
    }
  }
  
  /// Splits the tensor into chunks of size `into` along the specified axis.
  ///
  /// The last chunk may be smaller than `into` if the dimension is not evenly divisible.
  ///
  /// - Parameters:
  ///   - into: The maximum number of slices per chunk along the chosen axis.
  ///   - axis: The axis along which to split (`0` = rows, `1` = columns, `2` = depth). Defaults to `2`.
  /// - Returns: An array of tensors, each covering one chunk along `axis`.
  func split(into: Int, axis: Int = 2) -> [Tensor] {
    let columns = size.columns
    let rows = size.rows
    let depth = size.depth
    let selfPtr = storage.pointer
    let sliceSize = rows * columns
    
    if axis == 2 {
      let chunkCount = (depth + into - 1) / into
      var results: [Tensor] = []
      results.reserveCapacity(chunkCount)
      
      for chunk in 0..<chunkCount {
        let dStart = chunk * into
        let dEnd = min(dStart + into, depth)
        let chunkDepth = dEnd - dStart
        let newSize = TensorSize(rows: rows, columns: columns, depth: chunkDepth)
        let chunkStorage = TensorStorage.create(count: sliceSize * chunkDepth)
        let dstPtr = chunkStorage.pointer
        
        for d in 0..<chunkDepth {
          let srcStart = flatIndex(column: 0, row: 0, depth: dStart + d)
          let copyCount = min(sliceSize, storage.count - srcStart)
          if copyCount > 0 {
            (dstPtr + d * sliceSize).update(from: selfPtr + srcStart, count: copyCount)
          }
        }
        
        results.append(Tensor(storage: chunkStorage, size: newSize))
      }
      return results
      
    } else if axis == 0 {
      let chunkCount = (rows + into - 1) / into
      var results: [Tensor] = []
      results.reserveCapacity(chunkCount)
      
      for chunk in 0..<chunkCount {
        let rStart = chunk * into
        let rEnd = min(rStart + into, rows)
        let chunkRows = rEnd - rStart
        let newSize = TensorSize(rows: chunkRows, columns: columns, depth: depth)
        let chunkStorage = TensorStorage.create(count: columns * chunkRows * depth)
        let dstPtr = chunkStorage.pointer
        
        for d in 0..<depth {
          for r in 0..<chunkRows {
            let srcStart = flatIndex(column: 0, row: rStart + r, depth: d)
            let dstStart = d * chunkRows * columns + r * columns
            let copyCount = min(columns, storage.count - srcStart)
            if copyCount > 0 {
              (dstPtr + dstStart).update(from: selfPtr + srcStart, count: copyCount)
            }
          }
        }
        
        results.append(Tensor(storage: chunkStorage, size: newSize))
      }
      return results
      
    } else if axis == 1 {
      let chunkCount = (columns + into - 1) / into
      var results: [Tensor] = []
      results.reserveCapacity(chunkCount)
      
      for chunk in 0..<chunkCount {
        let cStart = chunk * into
        let cEnd = min(cStart + into, columns)
        let chunkCols = cEnd - cStart
        let newSize = TensorSize(rows: rows, columns: chunkCols, depth: depth)
        let chunkStorage = TensorStorage.create(count: chunkCols * rows * depth)
        let dstPtr = chunkStorage.pointer
        
        for d in 0..<depth {
          for r in 0..<rows {
            let dstStart = d * rows * chunkCols + r * chunkCols
            for c in 0..<chunkCols {
              let srcIdx = flatIndex(column: cStart + c, row: r, depth: d)
              dstPtr[dstStart + c] = srcIdx < storage.count ? selfPtr[srcIdx] : 0
            }
          }
        }
        
        results.append(Tensor(storage: chunkStorage, size: newSize))
      }
      return results
      
    } else {
      return [self]
    }
  }
  
  /// Returns a new tensor with element-wise square root, optionally adding a stability offset first.
  ///
  /// - Parameter adding: A small constant added to each element before taking the square root,
  ///   to avoid numerical instability near zero. Defaults to `Tensor.Scalar.stabilityFactor`.
  /// - Returns: A new `Tensor` with values `sqrt(element + adding)`.
  func sqrt(adding: Tensor.Scalar = .stabilityFactor) -> Tensor {
    let shifted = storage + adding
    let result = shifted.squareRoot()
    return Tensor(storage: result, size: size, context: context)
  }
  
  /// Returns the variance of elements, either globally or along a specific axis.
  ///
  /// - Parameter axis: The axis along which to compute variance. Pass `-1` (default) to
  ///   compute the global variance as a scalar tensor.
  /// - Returns: A `Tensor` containing the variance result.
  func variance(axis: Int = -1) -> Tensor {
    if axis == -1 {
      let meanVal = storage.mean
      let centered = storage - meanVal
      let sumSq = centered.sumOfSquares
      return Tensor(sumSq / Scalar(storage.count))
    }
    
    return apply(axis: axis) { ptr, count in
      let mean = NumSwiftFlat.mean(ptr, count: count)
      var sumSq: Tensor.Scalar = 0
      for i in 0..<count {
        let diff = ptr[i] - mean
        sumSq += diff * diff
      }
      return sumSq / Tensor.Scalar(count)
    }
  }
  
  /// Returns the mean of elements, either globally or along a specific axis.
  ///
  /// - Parameter axis: The axis along which to compute the mean. Pass `-1` (default) to
  ///   reduce all elements to a scalar tensor.
  /// - Returns: A `Tensor` containing the mean result.
  func mean(axis: Int = -1) -> Tensor {
    if axis == -1 {
      guard !storage.isEmpty else { return Tensor(Scalar(0)) }
      return Tensor(storage.mean)
    }
    
    return apply(axis: axis) { ptr, count in
      NumSwiftFlat.mean(ptr, count: count)
    }
  }
  
  /// Returns the sum of elements, either globally or along a specific axis.
  ///
  /// - Parameter axis: The axis along which to sum. Pass `-1` (default) to reduce
  ///   all elements to a scalar tensor.
  /// - Returns: A `Tensor` containing the sum result.
  func sum(axis: Int = -1) -> Tensor {
    if axis == -1 {
      return Tensor(storage.sum)
    } else {
      return apply(axis: axis) { ptr, count in
        NumSwiftFlat.sum(ptr, count: count)
      }
    }
  }
  
  /// Returns the cumulative subtraction of elements, either globally or along a specific axis.
  ///
  /// Starts from zero and subtracts each element in order. For global reduction (`axis == -1`)
  /// this is equivalent to the negated sum of all elements.
  ///
  /// - Parameter axis: The axis along which to subtract. Pass `-1` (default) for a global result.
  /// - Returns: A `Tensor` containing the subtraction result.
  func subtract(axis: Int = -1) -> Tensor {
    if axis == -1 {
      guard storage.count > 0 else { return Tensor(Scalar(0)) }
      var result: Scalar = 0
      for i in 0..<storage.count {
        result -= storage[i]
      }
      return Tensor(result)
    } else {
      return apply(axis: axis) { ptr, count in
        guard count > 0 else { return 0 }
        var result = ptr[0]
        for i in 1..<count {
          result -= ptr[i]
        }
        return result
      }
    }
  }
  
  /// Returns the product of all elements, either globally or along a specific axis.
  ///
  /// - Parameter axis: The axis along which to compute the product. Pass `-1` (default) to
  ///   reduce all elements to a scalar tensor.
  /// - Returns: A `Tensor` containing the product result.
  func multiply(axis: Int = -1) -> Tensor {
    if axis == -1 {
      guard storage.count > 0 else { return Tensor(Scalar(1)) }
      var result: Scalar = 1
      for i in 0..<storage.count {
        result *= storage[i]
      }
      return Tensor(result)
    } else {
      return apply(axis: axis) { ptr, count in
        var result: Tensor.Scalar = 1
        for i in 0..<count {
          result *= ptr[i]
        }
        return result
      }
    }
  }
  
  /// Returns the L2 norm (Euclidean length) of elements, either globally or along a specific axis.
  ///
  /// - Parameter axis: The axis along which to compute the norm. Pass `-1` (default) to
  ///   compute the global norm as a scalar tensor.
  /// - Returns: A `Tensor` containing the norm result.
  func norm(axis: Int = -1) -> Tensor {
    if axis == -1 {
      return Tensor(Tensor.Scalar.sqrt(storage.sumOfSquares))
    }
    
    return apply(axis: axis) { ptr, count in
      Tensor.Scalar.sqrt(NumSwiftFlat.sumOfSquares(ptr, count: count))
    }
  }
  
  /// Concatenates another tensor to this tensor along the specified axis.
  ///
  /// Axis semantics:
  /// - `0`: concatenate along rows (new rows below existing rows)
  /// - `1`: concatenate along columns (new columns to the right)
  /// - `2`: concatenate along depth
  /// - `3`: concatenate along the batch dimension (both tensors must share the same unit size)
  /// - `-1`: flat concatenation into a 1D tensor
  ///
  /// - Parameters:
  ///   - tensor: The tensor to append.
  ///   - axis: The dimension along which to concatenate. Defaults to `1` (columns).
  /// - Returns: A new `Tensor` that is the concatenation of `self` and `tensor`.
  @discardableResult
  func concat(_ tensor: Tensor, axis: Int = 1) -> Tensor {
    if isEmpty {
      return Tensor(storage: tensor.storage.copy(), size: tensor.size, context: context)
    }
    if tensor.isEmpty {
      return self
    }
    
    let selfCols = size.columns
    let selfRows = size.rows
    let selfDepth = size.depth
    let otherCols = tensor.size.columns
    let otherRows = tensor.size.rows
    let otherDepth = tensor.size.depth
    
    if axis == -1 {
      let totalCols = storage.count + tensor.storage.count
      let result = TensorStorage.create(count: totalCols)
      if storage.count > 0 {
        result.pointer.update(from: storage.pointer, count: storage.count)
      }
      if tensor.storage.count > 0 {
        (result.pointer + storage.count).update(from: tensor.storage.pointer, count: tensor.storage.count)
      }
      return Tensor(storage: result, size: TensorSize(rows: 1, columns: totalCols, depth: 1), context: context)
    }
    
    // reserved for creating a single Tensor with multiple batches.
    // Good for GPU parallization
    if axis == 3 {
      guard size.unitSize == tensor.size.unitSize else {
        assertionFailure("When adding along batch axis, all tensors must have the same unit size")
        return Tensor()
      }
      
      let newSize = TensorSize(rows: selfRows,
                               columns: selfCols,
                               depth: selfDepth,
                               batchCount: size.batchCount + tensor.size.batchCount)
      
      let result = TensorStorage.create(count: storage.count + tensor.storage.count)
      if storage.count > 0 {
        result.pointer.update(from: storage.pointer, count: storage.count)
      }
      if tensor.storage.count > 0 {
        (result.pointer + storage.count).update(from: tensor.storage.pointer, count: tensor.storage.count)
      }
      return Tensor(storage: result, size: newSize, context: context)
      
    } else if axis == 2 {
      let newDepth = selfDepth + otherDepth
      let newSize = TensorSize(rows: selfRows, columns: selfCols, depth: newDepth)
      let result = TensorStorage.create(count: storage.count + tensor.storage.count)
      if storage.count > 0 {
        result.pointer.update(from: storage.pointer, count: storage.count)
      }
      if tensor.storage.count > 0 {
        (result.pointer + storage.count).update(from: tensor.storage.pointer, count: tensor.storage.count)
      }
      return Tensor(storage: result, size: newSize, context: context)

    } else if axis == 0 {
      // Concat along rows
      let newRows = selfRows + otherRows
      let newSize = TensorSize(rows: newRows, columns: selfCols, depth: selfDepth)
      let result = TensorStorage.create(count: selfCols * newRows * selfDepth)
      let dstPtr = result.pointer
      let selfPtr = storage.pointer
      let otherPtr = tensor.storage.pointer
      
      for d in 0..<selfDepth {
        for r in 0..<selfRows {
          let srcStart = flatIndex(column: 0, row: r, depth: d)
          let dstStart = d * newRows * selfCols + r * selfCols
          (dstPtr + dstStart).update(from: selfPtr + srcStart, count: selfCols)
        }
        let minOtherRows = min(otherRows, (d < otherDepth) ? otherRows : 0)
        for r in 0..<minOtherRows {
          let srcStart = tensor.flatIndex(column: 0, row: r, depth: d)
          let dstStart = d * newRows * selfCols + (selfRows + r) * selfCols
          let colsToCopy = min(otherCols, selfCols)
          (dstPtr + dstStart).update(from: otherPtr + srcStart, count: colsToCopy)
        }
      }
      
      return Tensor(storage: result, size: newSize, context: context)
      
    } else if axis == 1 {
      // Concat along columns
      let newCols = selfCols + otherCols
      let newSize = TensorSize(rows: selfRows, columns: newCols, depth: selfDepth)
      let result = TensorStorage.create(count: newCols * selfRows * selfDepth)
      let dstPtr = result.pointer
      let selfPtr = storage.pointer
      let otherPtr = tensor.storage.pointer
      
      for d in 0..<selfDepth {
        for r in 0..<selfRows {
          let dstStart = d * selfRows * newCols + r * newCols
          let srcSelfStart = flatIndex(column: 0, row: r, depth: d)
          (dstPtr + dstStart).update(from: selfPtr + srcSelfStart, count: selfCols)
          if d < otherDepth && r < otherRows {
            let srcOtherStart = tensor.flatIndex(column: 0, row: r, depth: d)
            (dstPtr + dstStart + selfCols).update(from: otherPtr + srcOtherStart, count: otherCols)
          }
        }
      }
      
      return Tensor(storage: result, size: newSize, context: context)
    }
    
    return Tensor(storage: storage.copy(), size: size, context: context)
  }
  
  /// Returns a new tensor with its elements scaled to unit L2 norm.
  ///
  /// Divides every element by the Euclidean norm of the storage, without a stability offset.
  ///
  /// - Returns: A new `Tensor` normalized to unit length.
  func l2Normalized() -> Tensor {
    let sumSq = storage.sumOfSquares
    let divisor = Scalar.sqrt(sumSq)
    let result = storage / divisor
    return Tensor(storage: result, size: size, context: context)
  }
  
  /// Returns a new tensor by applying `transform` to every scalar element.
  ///
  /// - Parameter transform: A closure that maps each `Tensor.Scalar` to a new `Tensor.Scalar`.
  /// - Returns: A new `Tensor` with the same shape and context, containing the transformed values.
  func map(_ transform: (Tensor.Scalar) -> Tensor.Scalar) -> Tensor {
    let result = TensorStorage.create(count: storage.count)
    let srcPtr = storage.pointer
    let dstPtr = result.pointer
    for i in 0..<storage.count {
      dstPtr[i] = transform(srcPtr[i])
    }
    return Tensor(storage: result, size: size, context: context)
  }
  
  static func /(lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(storage: lhs / rhs.storage, size: rhs.size, context: rhs.context)
  }
  
  static func *(lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(storage: rhs.storage * lhs, size: rhs.size, context: rhs.context)
  }
  
  static func -(lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(storage: lhs - rhs.storage, size: rhs.size, context: rhs.context)
  }
  
  static func /(lhs: Tensor, rhs: Scalar) -> Tensor {
    return Tensor(storage: lhs.storage / rhs, size: lhs.size, context: lhs.context)
  }
  
  static func *(lhs: Tensor, rhs: Scalar) -> Tensor {
    return Tensor(storage: lhs.storage * rhs, size: lhs.size, context: lhs.context)
  }
  
  static func -(lhs: Tensor, rhs: Scalar) -> Tensor {
    return Tensor(storage: lhs.storage - rhs, size: lhs.size, context: lhs.context)
  }
  
  static func +(lhs: Tensor, rhs: Scalar) -> Tensor {
    return Tensor(storage: lhs.storage + rhs, size: lhs.size, context: lhs.context)
  }
  
  /// Performs element-wise addition between two tensors with automatic broadcasting support.
  static func +(lhs: Tensor, rhs: Tensor) -> Tensor {
    if let axis = Tensor.axisToApplyAlong(selfSize: lhs.size,
                                          size: rhs.size) {
      return lhs.addAlong(axis: axis, value: rhs)
    }
    
    if let new = lhs.broadcastPerChannelFastPath(value: rhs, op: .add) {
      new.label = "addition"
      if lhs.graphChain.contains(rhs.id) { new.setGraphSafe(lhs); new.setGraphSafe(rhs) }
      else { new.setGraphSafe(rhs); new.setGraphSafe(lhs) }
      return new
    }
    
    let result = lhs.storage + rhs.storage
    
    let new = Tensor(storage: result, size: lhs.size, context: lhs.addContext(value: rhs))
    new.label = "addition"
    
    if lhs.graphChain.contains(rhs.id) {
      new.setGraphSafe(lhs)
      new.setGraphSafe(rhs)
    } else {
      new.setGraphSafe(rhs)
      new.setGraphSafe(lhs)
    }
    
    return new
  }
  
  /// Performs element-wise subtraction between two tensors with automatic broadcasting support.
  static func -(lhs: Tensor, rhs: Tensor) -> Tensor {
    if let axis = Tensor.axisToApplyAlong(selfSize: lhs.size,
                                          size: rhs.size) {
      return lhs.subtractAlong(axis: axis, value: rhs)
    }
    
    if let new = lhs.broadcastPerChannelFastPath(value: rhs, op: .sub) {
      new.label = "subtraction"
      if lhs.graphChain.contains(rhs.id) { new.setGraphSafe(lhs); new.setGraphSafe(rhs) }
      else { new.setGraphSafe(rhs); new.setGraphSafe(lhs) }
      return new
    }
    
    let result = lhs.storage - rhs.storage

    let new = Tensor(storage: result, size: lhs.size, context: lhs.subtractContext(value: rhs))
    new.label = "subtraction"

    if lhs.graphChain.contains(rhs.id) {
      new.setGraphSafe(lhs)
      new.setGraphSafe(rhs)
    } else {
      new.setGraphSafe(rhs)
      new.setGraphSafe(lhs)
    }
    
    return new
  }
  
  /// Performs element-wise multiplication between two tensors with automatic broadcasting support.
  static func *(lhs: Tensor, rhs: Tensor) -> Tensor {
    if let axis = Tensor.axisToApplyAlong(selfSize: lhs.size,
                                          size: rhs.size) {
      return lhs.multiplyAlong(axis: axis, value: rhs)
    }
    
    if let new = lhs.broadcastPerChannelFastPath(value: rhs, op: .mul) {
      new.label = "multiplication"
      
      if lhs.graphChain.contains(rhs.id) {
        new.setGraphSafe(lhs)
        new.setGraphSafe(rhs)
      } else {
        new.setGraphSafe(rhs)
        new.setGraphSafe(lhs)
      }
      
      return new
    }
    
    let result = lhs.storage * rhs.storage

    let new = Tensor(storage: result, size: lhs.size, context: lhs.multiplyContext(value: rhs))
    new.label = "multiplication"

    if lhs.graphChain.contains(rhs.id) {
      new.setGraphSafe(lhs)
      new.setGraphSafe(rhs)
    } else {
      new.setGraphSafe(rhs)
      new.setGraphSafe(lhs)
    }
    
    return new
  }
  
  /// Performs element-wise division between two tensors with automatic broadcasting support.
  static func /(lhs: Tensor, rhs: Tensor) -> Tensor {
    if let axis = Tensor.axisToApplyAlong(selfSize: lhs.size,
                                          size: rhs.size) {
      return lhs.divideAlong(axis: axis, value: rhs)
    }
    
    if let new = lhs.broadcastPerChannelFastPath(value: rhs, op: .div) {
      new.label = "division"
      if lhs.graphChain.contains(rhs.id) { new.setGraphSafe(lhs); new.setGraphSafe(rhs) }
      else { new.setGraphSafe(rhs); new.setGraphSafe(lhs) }
      return new
    }
    
    let result = lhs.storage / rhs.storage

    let new = Tensor(storage: result, size: lhs.size, context: lhs.divideContext(value: rhs))
    new.label = "division"

    if lhs.graphChain.contains(rhs.id) {
      new.setGraphSafe(lhs)
      new.setGraphSafe(rhs)
    } else {
      new.setGraphSafe(rhs)
      new.setGraphSafe(lhs)
    }
    
    return new
  }
  
  func zerosLike() -> Tensor {
    return Tensor(storage: TensorStorage.create(count: storage.count), size: size)
  }
  
  func onesLike() -> Tensor {
    return Tensor(storage: TensorStorage.create(repeating: 1, count: storage.count), size: size)
  }
  
  func transposed() -> Tensor {
    let columns = size.columns
    let rows = size.rows
    let depth = size.depth
    
    let newSize = TensorSize(rows: columns, columns: rows, depth: depth)
    let sliceSize = rows * columns
    
    if depth == 1 {
      let resultStorage = TensorStorage.create(count: storage.count)
      NumSwiftFlat.transpose(storage.pointer, result: resultStorage.pointer, rows: rows, columns: columns)
      return Tensor(storage: resultStorage, size: newSize, context: context)
    }
    
    let resultStorage = TensorStorage.create(count: storage.count)
    
    if sliceSize <= 2048 {
      for d in 0..<depth {
        let base = d * sliceSize
        for r in 0..<rows {
          for c in 0..<columns {
            let srcIdx = base + r * columns + c
            let dstIdx = base + c * rows + r
            resultStorage[dstIdx] = storage[srcIdx]
          }
        }
      }
    } else {
      for d in 0..<depth {
        let offset = d * sliceSize
        NumSwiftFlat.transpose(storage.pointer + offset, result: resultStorage.pointer + offset,
                               rows: rows, columns: columns)
      }
    }
    
    return Tensor(storage: resultStorage, size: newSize, context: context)
  }
}

// NOTE: debugDescription, Array<Tensor> extensions, fillRandom/fillWith, and Gradient operators
// have been moved to Tensor.swift as part of the flat storage refactor.

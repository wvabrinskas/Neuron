//
//  File.swift
//
//
//  Created by William Vabrinskas on 6/27/22.
//

import Foundation
import NumSwift

public extension Float {
  static var stabilityFactor: Self {
    1e-12
  }
}

#if arch(arm64)
public extension Float16 {
  static var stabilityFactor: Self {
    1e-4
  }
}
#endif

public extension Tensor {
  typealias MathBlock = (_ feature: [Scalar]) -> Scalar
  typealias MathAlongBlock = (_ feature: [Scalar], _ value: ([Scalar]?, Scalar?)) -> [Scalar]
  
  /*
      +--------+
     /        /|
    /        Z |
   +---X----+  |
   |        |  |
   |   -1   Y  +
   |        | /
   |        |/
   +--------+
   Along axis 0 the Tensor of shape AxBxC, where A is the columns, B is the rows, and C is the depth, would perform a mathematical function along the Y axis returning a (Ax1xC) Tensor
   
   Along axis 1 the Tensor of shape AxBxC, where A is the columns, B is the rows, and C is the depth, would perform a mathematical function along the X axis returning a (1xBxC) Tensor
   
   Along axis 2 the Tensor of shape AxBxC, where A is the columns, B is the rows, and C is the depth, would perform a mathematical function along the Z axis returning a (AxBx1) Tensor
   
   Along axis -1 the Tensor of shape AxBxC, where A is the columns, B is the rows, and C is the depth, would perform a mathematical function along the Z axis returning a (1x1x1) Tensor Scalar
   */
  func apply(axis: Int, _ block: MathBlock) -> Tensor {
    let columns = size.columns
    let rows = size.rows
    let depth = size.depth
    
    if axis == 0 {
      // Reduce along rows -> output (columns x 1 x depth)
      let outSize = TensorSize(rows: 1, columns: columns, depth: depth)
      var outStorage = Tensor.Value(repeating: 0, count: columns * depth)
      
      for d in 0..<depth {
        for c in 0..<columns {
          var workingRow = [Scalar]()
          workingRow.reserveCapacity(rows)
          for r in 0..<rows {
            workingRow.append(storage[flatIndex(column: c, row: r, depth: d)])
          }
          outStorage[d * columns + c] = block(workingRow)
        }
      }
      
      return Tensor(outStorage, size: outSize)
      
    } else if axis == 1 {
      // Reduce along columns -> output (1 x rows x depth)
      let outSize = TensorSize(rows: rows, columns: 1, depth: depth)
      var outStorage = Tensor.Value(repeating: 0, count: rows * depth)
      
      for d in 0..<depth {
        for r in 0..<rows {
          let start = flatIndex(column: 0, row: r, depth: d)
          let row = Array(storage[start..<(start + columns)])
          outStorage[d * rows + r] = block(row)
        }
      }
      
      return Tensor(outStorage, size: outSize)
      
    } else if axis == 2 {
      // Reduce along depth -> output (columns x rows x 1)
      let outSize = TensorSize(rows: rows, columns: columns, depth: 1)
      var outStorage = Tensor.Value(repeating: 0, count: columns * rows)
      
      for r in 0..<rows {
        for c in 0..<columns {
          var featureR = [Scalar]()
          featureR.reserveCapacity(depth)
          for d in 0..<depth {
            featureR.append(storage[flatIndex(column: c, row: r, depth: d)])
          }
          outStorage[r * columns + c] = block(featureR)
        }
      }
      
      return Tensor(outStorage, size: outSize)
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

  private enum _BroadcastOp { case add, sub, mul, div }

  /// Pointer-based fast path for *Along broadcasting. Returns result storage when applicable, nil to fall back.
  private func _broadcastAlongFastPath(axis: Int, value: Tensor, op: _BroadcastOp) -> Tensor? {
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
  
  /// Applies a mathematical operation along a specific axis with broadcasting support.
  /// This is the core function used by other *Along methods.
  ///
  /// - Parameters:
  ///   - axis: The axis along which to perform the operation (0, 1, or 2)
  ///   - input: The tensor to operate with, which will be broadcast along the specified axis
  ///   - block: The mathematical operation to apply
  /// - Returns: A new tensor with the result of the operation
  ///
  /// - Note: Self-assignment is supported. Methods using this function automatically detect and prevent
  ///   reference cycles in the computation graph via ` `.
  func applyAlong(axis: Int, input: Tensor, _ block: MathAlongBlock) -> Tensor {
    let inputSize = input.size
    let selfSize = self.size
    let columns = selfSize.columns
    let rows = selfSize.rows
    let depth = selfSize.depth
        
    var result: Tensor.Value = []
    result.reserveCapacity(selfSize.depth * selfSize.rows * selfSize.columns)
    
    for d in 0..<depth {
      for r in 0..<rows {
        // Extract the row from self
        let selfStart = flatIndex(column: 0, row: r, depth: d)
        let feature = storage[safe: selfStart..<(selfStart + columns), 0]
        
        let out: [Scalar]
        
        if axis == 0,
           inputSize.columns == columns,
           inputSize.rows == 1,
           inputSize.depth == depth {
          let inputStart = input.flatIndex(column: 0, row: 0, depth: d)
          let v = input.storage[safe: inputStart..<(inputStart + inputSize.columns), 0]
          out = block(feature, (v, nil))
          
        } else if axis == 1,
                  inputSize.columns == 1,
                  inputSize.rows == rows,
                  inputSize.depth == depth {
          let v = input.storage[safe: input.flatIndex(column: 0, row: r, depth: d), 0]
          out = block(feature, (nil, v))
          
        } else if axis == 2,
                  inputSize.columns == columns,
                  inputSize.rows == rows,
                  inputSize.depth == 1 {
          let inputStart = input.flatIndex(column: 0, row: r, depth: 0)
          let v = input.storage[safe: inputStart..<(inputStart + inputSize.columns), 0]
          out = block(feature, (v, nil))
        } else {
          out = feature
        }
        
        result.append(contentsOf: out)
      }
    }
    
    return Tensor(result, size: selfSize)
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
        let gradA = gradient * (1 / value)
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
    if let new = _broadcastAlongFastPath(axis: axis, value: value, op: .div) {
      new.label = "division"
      if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
      else { new.setGraphSafe(value); new.setGraphSafe(self) }
      return new
    }
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature / valueArray
      } else if let valueScalar = value.1 {
        return feature / valueScalar
      } else {
        return feature
      }
    }
    let out = applyAlong(axis: axis, input: value, block)
    let new = Tensor(storage: out.storage, size: out.size, context: divideContext(value: value))
    new.label = "division"
    if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
    else { new.setGraphSafe(value); new.setGraphSafe(self) }
    return new
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
    if let new = _broadcastAlongFastPath(axis: axis, value: value, op: .mul) {
      new.label = "multiplication"
      if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
      else { new.setGraphSafe(value); new.setGraphSafe(self) }
      return new
    }
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature * valueArray
      } else if let valueScalar = value.1 {
        return feature * valueScalar
      } else {
        return feature
      }
    }
    let out = applyAlong(axis: axis, input: value, block)
    let new = Tensor(storage: out.storage, size: out.size, context: multiplyContext(value: value))
    new.label = "multiplication"
    if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
    else { new.setGraphSafe(value); new.setGraphSafe(self) }
    return new
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
    if let new = _broadcastAlongFastPath(axis: axis, value: value, op: .add) {
      new.label = "addition"
      if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
      else { new.setGraphSafe(value); new.setGraphSafe(self) }
      return new
    }
    // Fallback: generic applyAlong path
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature + valueArray
      } else if let valueScalar = value.1 {
        return feature + valueScalar
      } else {
        return feature
      }
    }
    let out = applyAlong(axis: axis, input: value, block)
    
    let new = Tensor(storage: out.storage, size: out.size, context: addContext(value: value))
    
    new.label = "addition"

    if graphChain.contains(value.id) {
      // non branched node
      new.setGraphSafe(self)
      new.setGraphSafe(value)
    } else {
      // branched node
      new.setGraphSafe(value)
      new.setGraphSafe(self)
    }
    
    return new
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
    if let new = _broadcastAlongFastPath(axis: axis, value: value, op: .sub) {
      new.label = "subtraction"
      if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
      else { new.setGraphSafe(value); new.setGraphSafe(self) }
      return new
    }
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature - valueArray
      } else if let valueScalar = value.1 {
        return feature - valueScalar
      } else {
        return feature
      }
    }
    let out = applyAlong(axis: axis, input: value, block)
    let new = Tensor(storage: out.storage, size: out.size, context: subtractContext(value: value))
    new.label = "subtraction"
    if graphChain.contains(value.id) { new.setGraphSafe(self); new.setGraphSafe(value) }
    else { new.setGraphSafe(value); new.setGraphSafe(self) }
    return new
  }
  
  func sum() -> Scalar {
    storage.sum
  }
  
  func testLarge(limit: Scalar) {
    for val in storage {
      if val > limit {
        assertionFailure()
        return
      }
    }
  }
  
  func testInvalid() {
    for val in storage {
      if val.isNormal == false {
        print(self)
        fatalError()
        return
      }
    }
  }
  
  func testInf() {
    for val in storage {
      if val.isInfinite {
        assertionFailure()
        return
      }
    }
  }
  
  func testNaN() {
    for val in storage {
      if val.isNaN {
        assertionFailure()
        return
      }
    }
  }
  
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
  
  func sumOfSquares(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      feature.sumOfSquares
    }
    
    if axis == -1 {
      return Tensor(storage.sumOfSquares)
    }
    
    return apply(axis: axis, block)
  }
  
  func split(into: Int, axis: Int = 2) -> [Tensor] {
    let columns = size.columns
    let rows = size.rows
    let depth = size.depth
    
    if axis == 2 {
      let chunkCount = (depth + into - 1) / into
      var results: [Tensor] = []
      results.reserveCapacity(chunkCount)
      
      for chunk in 0..<chunkCount {
        let dStart = chunk * into
        let dEnd = min(dStart + into, depth)
        let chunkDepth = dEnd - dStart
        let newSize = TensorSize(rows: rows, columns: columns, depth: chunkDepth)
        var chunkStorage = Tensor.Value(repeating: 0, count: columns * rows * chunkDepth)
        
        for d in 0..<chunkDepth {
          let srcDepth = dStart + d
          let srcStart = flatIndex(column: 0, row: 0, depth: srcDepth)
          let dstStart = d * rows * columns
          for i in 0..<(rows * columns) {
            let srcIdx = srcStart + i
            chunkStorage[dstStart + i] = srcIdx < storage.count ? storage[srcIdx] : 0
          }
        }
        
        results.append(Tensor(chunkStorage, size: newSize))
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
        var chunkStorage = Tensor.Value(repeating: 0, count: columns * chunkRows * depth)
        
        for d in 0..<depth {
          for r in 0..<chunkRows {
            let srcStart = flatIndex(column: 0, row: rStart + r, depth: d)
            let dstStart = d * chunkRows * columns + r * columns
            for c in 0..<columns {
              let srcIdx = srcStart + c
              chunkStorage[dstStart + c] = srcIdx < storage.count ? storage[srcIdx] : 0
            }
          }
        }
        
        results.append(Tensor(chunkStorage, size: newSize))
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
        var chunkStorage = Tensor.Value(repeating: 0, count: chunkCols * rows * depth)
        
        for d in 0..<depth {
          for r in 0..<rows {
            let dstStart = d * rows * chunkCols + r * chunkCols
            for c in 0..<chunkCols {
              let srcIdx = flatIndex(column: cStart + c, row: r, depth: d)
              chunkStorage[dstStart + c] = srcIdx < storage.count ? storage[srcIdx] : 0
            }
          }
        }
        
        results.append(Tensor(chunkStorage, size: newSize))
      }
      return results
      
    } else {
      return [self]
    }
  }
  
  func sqrt(adding: Tensor.Scalar = .stabilityFactor) -> Tensor {
    let shifted = storage + adding
    let result = shifted.squareRoot()
    return Tensor(storage: result, size: size, context: context)
  }
  
  func variance(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      let mean = feature.mean
      let sumOSquares = (feature - mean).sumOfSquares
      
      let count = feature.count
      
      return sumOSquares / Tensor.Scalar(count)
    }
    
    if axis == -1 {
      let meanVal = storage.mean
      let centered = storage - meanVal
      let sumSq = centered.sumOfSquares
      return Tensor(sumSq / Scalar(storage.count))
    }
    
    return apply(axis: axis, block)
  }
  
  func mean(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      feature.mean
    }
    
    if axis == -1 {
      guard !storage.isEmpty else { return Tensor(Scalar(0)) }
      return Tensor(storage.mean)
    }
    
    return apply(axis: axis, block)
  }
  
  func sum(axis: Int = -1) -> Tensor {
    if axis == -1 {
      return Tensor(storage.sum)
    } else {
      return apply(axis: axis) { feature in
        feature.sum
      }
    }
  }
  
  func subtract(axis: Int = -1) -> Tensor {
    if axis == -1 {
      guard storage.count > 0 else { return Tensor(Scalar(0)) }
      var result: Scalar = 0
      for i in 0..<storage.count {
        result -= storage[i]
      }
      return Tensor(result)
    } else {
      return apply(axis: axis) { feature in
        var feature = feature
        let first = feature.first ?? 0
        feature = Array(feature.dropFirst())
        return feature.reduce(first, -)
      }
    }
  }
  
  func multiply(axis: Int = -1) -> Tensor {
    if axis == -1 {
      guard storage.count > 0 else { return Tensor(Scalar(1)) }
      var result: Scalar = 1
      for i in 0..<storage.count {
        result *= storage[i]
      }
      return Tensor(result)
    } else {
      return apply(axis: axis) { feature in
        feature.reduce(1, *)
      }
    }
  }
  
  func norm(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      Tensor.Scalar.sqrt(feature.sumOfSquares)
    }
    
    if axis == -1 {
      return Tensor(Tensor.Scalar.sqrt(storage.sumOfSquares))
    }
    
    return apply(axis: axis, block)
  }
  
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
    
    if axis == 2 {
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
      var result = Tensor.Value(repeating: 0, count: selfCols * newRows * selfDepth)
      
      for d in 0..<selfDepth {
        for r in 0..<selfRows {
          let srcStart = flatIndex(column: 0, row: r, depth: d)
          let dstStart = d * newRows * selfCols + r * selfCols
          for c in 0..<selfCols {
            result[dstStart + c] = storage[srcStart + c]
          }
        }
        let minOtherRows = min(otherRows, (d < otherDepth) ? otherRows : 0)
        for r in 0..<minOtherRows {
          let srcStart = tensor.flatIndex(column: 0, row: r, depth: d)
          let dstStart = d * newRows * selfCols + (selfRows + r) * selfCols
          let colsToCopy = min(otherCols, selfCols)
          for c in 0..<colsToCopy {
            result[dstStart + c] = tensor.storage[srcStart + c]
          }
        }
      }
      
      return Tensor(result, size: newSize, context: context)
      
    } else if axis == 1 {
      // Concat along columns
      let newCols = selfCols + otherCols
      let newSize = TensorSize(rows: selfRows, columns: newCols, depth: selfDepth)
      var result = Tensor.Value(repeating: 0, count: newCols * selfRows * selfDepth)
      
      for d in 0..<selfDepth {
        for r in 0..<selfRows {
          let dstStart = d * selfRows * newCols + r * newCols
          // Copy self row
          let srcSelfStart = flatIndex(column: 0, row: r, depth: d)
          for c in 0..<selfCols {
            result[dstStart + c] = storage[srcSelfStart + c]
          }
          // Copy other row
          if d < otherDepth && r < otherRows {
            let srcOtherStart = tensor.flatIndex(column: 0, row: r, depth: d)
            for c in 0..<otherCols {
              result[dstStart + selfCols + c] = tensor.storage[srcOtherStart + c]
            }
          }
        }
      }
      
      return Tensor(result, size: newSize, context: context)
    }
    
    return Tensor(storage: storage.copy(), size: size, context: context)
  }
  
  func l2Normalized() -> Tensor {
    let sumSq = storage.sumOfSquares
    let divisor = Scalar.sqrt(sumSq)
    let result = storage / divisor
    return Tensor(storage: result, size: size, context: context)
  }
  
  func map(_ transform: (Tensor.Scalar) -> Tensor.Scalar) -> Tensor {
    var result = Tensor.Value(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      result[i] = transform(storage[i])
    }
    return Tensor(result, size: size, context: context)
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
    
    let result = lhs.storage + rhs.storage
    
    let new = Tensor(storage: result, size: lhs.size, context: lhs.addContext(value: rhs))
    new.label = "addition"
    
    if lhs.graphChain.contains(rhs.id) {
      // non branched node
      new.setGraphSafe(lhs)
      new.setGraphSafe(rhs)
    } else {
      // branched node
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
    
    let result = lhs.storage - rhs.storage

    let new = Tensor(storage: result, size: lhs.size, context: lhs.subtractContext(value: rhs))
    new.label = "subtraction"

    if lhs.graphChain.contains(rhs.id) {
      // non branched node
      new.setGraphSafe(lhs)
      new.setGraphSafe(rhs)
    } else {
      // branched node
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
    
    let result = lhs.storage * rhs.storage

    let new = Tensor(storage: result, size: lhs.size, context: lhs.multiplyContext(value: rhs))
    new.label = "multiplication"

    if lhs.graphChain.contains(rhs.id) {
      // non branched node
      new.setGraphSafe(lhs)
      new.setGraphSafe(rhs)
    } else {
      // branched node
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
    
    let result = lhs.storage / rhs.storage

    let new = Tensor(storage: result, size: lhs.size, context: lhs.divideContext(value: rhs))
    new.label = "division"

    if lhs.graphChain.contains(rhs.id) {
      // non branched node
      new.setGraphSafe(lhs)
      new.setGraphSafe(rhs)
    } else {
      // branched node
      new.setGraphSafe(rhs)
      new.setGraphSafe(lhs)
    }
    
    return new
  }
  
  func zerosLike() -> Tensor {
    let zeroStorage = Tensor.Value(repeating: 0, count: storage.count)
    return Tensor(zeroStorage, size: size)
  }
  
  func onesLike() -> Tensor {
    let oneStorage = Tensor.Value(repeating: 1, count: storage.count)
    return Tensor(oneStorage, size: size)
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

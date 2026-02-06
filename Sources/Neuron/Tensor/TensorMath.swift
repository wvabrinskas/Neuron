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
    
    return Tensor(storage, size: size)
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
    
    var outStorage = Tensor.Value(repeating: 0, count: storage.count)
    
    for d in 0..<depth {
      for r in 0..<rows {
        // Extract the row from self
        let selfStart = flatIndex(column: 0, row: r, depth: d)
        let feature = Array(storage[selfStart..<(selfStart + columns)])
        
        let out: [Scalar]
        
        if axis == 0,
           inputSize.columns == columns,
           inputSize.rows == 1,
           inputSize.depth == depth {
          // Broadcasting along rows: input has 1 row, broadcast across all rows
          let inputStart = input.flatIndex(column: 0, row: 0, depth: d)
          let v = Array(input.storage[inputStart..<(inputStart + inputSize.columns)])
          out = block(feature, (v, nil))
          
        } else if axis == 1,
                  inputSize.columns == 1,
                  inputSize.rows == rows,
                  inputSize.depth == depth {
          // Broadcasting along columns: input has 1 column, broadcast across all columns
          let v = input.storage[input.flatIndex(column: 0, row: r, depth: d)]
          out = block(feature, (nil, v))
          
        } else if axis == 2,
                  inputSize.columns == columns,
                  inputSize.rows == rows,
                  inputSize.depth == 1 {
          // Broadcasting along depth: input has depth=1, broadcast across all depth
          let inputStart = input.flatIndex(column: 0, row: r, depth: 0)
          let v = Array(input.storage[inputStart..<(inputStart + inputSize.columns)])
          out = block(feature, (v, nil))
        } else {
          out = feature
        }
        
        let outStart = d * rows * columns + r * columns
        for c in 0..<out.count {
          outStorage[outStart + c] = out[c]
        }
      }
    }
    
    return Tensor(outStorage, size: selfSize)
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
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature / valueArray
      } else if let valueScalar = value.1 {
        return feature / valueScalar
      } else {
        return feature
      }
    }
    
    let copied = value.copy()
    
    let context = TensorContext { inputs, gradient, wrt in
      if let wrt, (value.graphChain.contains(wrt.id) || value.id == wrt.id) {
        
        let result = gradient * (-1 * (inputs / (copied * copied)))
        
        return (result, Tensor(), Tensor())
      }

      return (gradient * (1 / copied), Tensor(), Tensor())
    }
    
    let out = applyAlong(axis: axis, input: value, block)
    
    let new = Tensor(out.storage, size: out.size, context: context)
    
    new.label = "division"
    
    // TODO: Somewhere something is calling this on divide or mult or add or sub and causing a memory leak...
    new.setGraphSafe(self)
    new.setGraphSafe(value)
    
    return new
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
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature * valueArray
      } else if let valueScalar = value.1 {
        return feature * valueScalar
      } else {
        return feature
      }
    }
    
    let copied = value.copy()

    let context = TensorContext { inputs, gradient, wrt in
      return (gradient * copied, Tensor(), Tensor())
    }
    
    let out = applyAlong(axis: axis, input: value, block)
    
    let new = Tensor(out.storage, size: out.size, context: context)
    
    new.label = "multiplication"

    new.setGraphSafe(self)
    new.setGraphSafe(value)
    
    return new
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
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature + valueArray
      } else if let valueScalar = value.1 {
        return feature + valueScalar
      } else {
        return feature
      }
    }
    
    let context = TensorContext { inputs, gradient, wrt in
      let copy = gradient.copy()
      copy.label = "addition"
      return (copy, Tensor(), Tensor())
    }
    
    let out = applyAlong(axis: axis, input: value, block)
    
    let new = Tensor(out.storage, size: out.size, context: context)
    
    new.label = "addition"

    new.setGraphSafe(self)
    new.setGraphSafe(value)
    
    return new
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
    let block: MathAlongBlock = { feature, value in
      if let valueArray = value.0 {
        return feature - valueArray
      } else if let valueScalar = value.1 {
        return feature - valueScalar
      } else {
        return feature
      }
    }
    
    let context = TensorContext { inputs, gradient, wrt in
      if let wrt, (value.graphChain.contains(wrt.id) || value.id == wrt.id) {
        return (gradient * -1, Tensor(), Tensor())
      }

      return (gradient, Tensor(), Tensor())
    }
    
    let out = applyAlong(axis: axis, input: value, block)
    
    let new = Tensor(out.storage, size: out.size, context: context)
    
    new.label = "subtraction"
    
    new.setGraphSafe(self)
    new.setGraphSafe(value)
    
    return new
  }
  
  func sum() -> Scalar {
    return NumSwiftFlat.sum(storage)
  }
  
  func testLarge(limit: Scalar) {
    for val in storage {
      if val > limit {
        assertionFailure()
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
    
    var result = Tensor.Value(repeating: 0, count: cSliceSize * depth)
    
    for d in 0..<depth {
      let aStart = d * aSliceSize
      let bStart = d * bSliceSize
      let cStart = d * cSliceSize
      
      let aSlice = Tensor.Value(storage[aStart..<(aStart + aSliceSize)])
      let bSlice = Tensor.Value(with.storage[bStart..<(bStart + bSliceSize)])
      
      let cSlice = NumSwiftFlat.matmul(aSlice, bSlice,
                                        aRows: aRows, aCols: aCols,
                                        bRows: bRows, bCols: bCols)
      
      for i in 0..<cSliceSize {
        result[cStart + i] = cSlice[i]
      }
    }
    
    let outSize = TensorSize(rows: aRows, columns: bCols, depth: depth)
    return Tensor(result, size: outSize)
  }
  
  func sumOfSquares(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      feature.sumOfSquares
    }
    
    if axis == -1 {
      return Tensor(NumSwiftFlat.sumOfSquares(storage))
    }
    
    return apply(axis: axis, block)
  }
  
  func split(into: Int, axis: Int = 2) -> [Tensor] {
    let columns = size.columns
    let rows = size.rows
    let depth = size.depth
    
    if axis == 2 {
      // Split along depth into groups of `into`
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
          let srcStart = srcDepth * rows * columns
          let dstStart = d * rows * columns
          for i in 0..<(rows * columns) {
            chunkStorage[dstStart + i] = storage[srcStart + i]
          }
        }
        
        results.append(Tensor(chunkStorage, size: newSize))
      }
      return results
      
    } else if axis == 0 {
      // Split along rows into groups of `into`
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
              chunkStorage[dstStart + c] = storage[srcStart + c]
            }
          }
        }
        
        results.append(Tensor(chunkStorage, size: newSize))
      }
      return results
      
    } else if axis == 1 {
      // Split along columns into groups of `into`
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
              chunkStorage[dstStart + c] = storage[flatIndex(column: cStart + c, row: r, depth: d)]
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
    let shifted = NumSwiftFlat.add(storage, scalar: adding)
    let result = NumSwiftFlat.sqrt(shifted)
    return Tensor(result, size: size, context: context)
  }
  
  func variance(axis: Int = -1) -> Tensor {
    let block: MathBlock = { feature in
      let mean = feature.mean
      let sumOSquares = (feature - mean).sumOfSquares
      
      let count = feature.count
      
      return sumOSquares / Tensor.Scalar(count)
    }
    
    if axis == -1 {
      let meanVal = NumSwiftFlat.mean(storage)
      let centered = NumSwiftFlat.subtract(storage, scalar: meanVal)
      let sumSq = NumSwiftFlat.sumOfSquares(centered)
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
      return Tensor(NumSwiftFlat.mean(storage))
    }
    
    return apply(axis: axis, block)
  }
  
  func sum(axis: Int = -1) -> Tensor {
    if axis == -1 {
      return Tensor(NumSwiftFlat.sum(storage))
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
      return Tensor(Tensor.Scalar.sqrt(NumSwiftFlat.sumOfSquares(storage)))
    }
    
    return apply(axis: axis, block)
  }
  
  @discardableResult
  func concat(_ tensor: Tensor, axis: Int = 1) -> Tensor {
    // Handle empty tensors
    if isEmpty {
      return Tensor(Tensor.Value(tensor.storage), size: tensor.size, context: context)
    }
    if tensor.isEmpty {
      return Tensor(Tensor.Value(storage), size: size, context: context)
    }
    
    let selfCols = size.columns
    let selfRows = size.rows
    let selfDepth = size.depth
    let otherCols = tensor.size.columns
    let otherRows = tensor.size.rows
    let otherDepth = tensor.size.depth
    
    if axis == -1 {
      // Flatten concat
      var result = storage
      result.append(contentsOf: tensor.storage)
      let totalCols = storage.count + tensor.storage.count
      return Tensor(result, size: TensorSize(rows: 1, columns: totalCols, depth: 1), context: context)
    }
    
    if axis == 2 {
      // Concat along depth
      let newDepth = selfDepth + otherDepth
      
      if selfRows == otherRows && selfCols == otherCols {
        // Fast path: same spatial dimensions, just append depth slices
        let newSize = TensorSize(rows: selfRows, columns: selfCols, depth: newDepth)
        var result = storage
        result.append(contentsOf: tensor.storage)
        return Tensor(result, size: newSize, context: context)
      } else {
        // Ragged concat: different spatial dims per depth slice, normalize via max dims
        let maxRows = max(selfRows, otherRows)
        let maxCols = max(selfCols, otherCols)
        let newSize = TensorSize(rows: maxRows, columns: maxCols, depth: newDepth)
        var result = Tensor.Value(repeating: 0, count: maxCols * maxRows * newDepth)
        
        // Copy self depth slices
        for d in 0..<selfDepth {
          for r in 0..<selfRows {
            for c in 0..<selfCols {
              let srcIdx = d * selfRows * selfCols + r * selfCols + c
              let dstIdx = d * maxRows * maxCols + r * maxCols + c
              result[dstIdx] = storage[srcIdx]
            }
          }
        }
        
        // Copy other depth slices
        for d in 0..<otherDepth {
          for r in 0..<otherRows {
            for c in 0..<otherCols {
              let srcIdx = d * otherRows * otherCols + r * otherCols + c
              let dstIdx = (selfDepth + d) * maxRows * maxCols + r * maxCols + c
              result[dstIdx] = tensor.storage[srcIdx]
            }
          }
        }
        
        return Tensor(result, size: newSize, context: context)
      }
      
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
    
    return Tensor(Tensor.Value(storage), size: size, context: context)
  }
  
  func l2Normalized() -> Tensor {
    let sumSq = NumSwiftFlat.sumOfSquares(storage)
    let divisor = Scalar.sqrt(sumSq)
    let result = NumSwiftFlat.divide(storage, scalar: divisor)
    return Tensor(result, size: size, context: context)
  }
  
  func map(_ transform: (Tensor.Scalar) -> Tensor.Scalar) -> Tensor {
    var result = Tensor.Value(repeating: 0, count: storage.count)
    for i in 0..<storage.count {
      result[i] = transform(storage[i])
    }
    return Tensor(result, size: size, context: context)
  }
  
  static func /(lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(NumSwiftFlat.divide(scalar: lhs, rhs.storage), size: rhs.size, context: rhs.context)
  }
  
  static func *(lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(NumSwiftFlat.multiply(rhs.storage, scalar: lhs), size: rhs.size, context: rhs.context)
  }
  
  static func -(lhs: Scalar, rhs: Tensor) -> Tensor {
    return Tensor(NumSwiftFlat.subtract(scalar: lhs, rhs.storage), size: rhs.size, context: rhs.context)
  }
  
  static func /(lhs: Tensor, rhs: Scalar) -> Tensor {
    return Tensor(NumSwiftFlat.divide(lhs.storage, scalar: rhs), size: lhs.size, context: lhs.context)
  }
  
  static func *(lhs: Tensor, rhs: Scalar) -> Tensor {
    return Tensor(NumSwiftFlat.multiply(lhs.storage, scalar: rhs), size: lhs.size, context: lhs.context)
  }
  
  static func -(lhs: Tensor, rhs: Scalar) -> Tensor {
    return Tensor(NumSwiftFlat.subtract(lhs.storage, scalar: rhs), size: lhs.size, context: lhs.context)
  }
  
  static func +(lhs: Tensor, rhs: Scalar) -> Tensor {
    return Tensor(NumSwiftFlat.add(lhs.storage, scalar: rhs), size: lhs.size, context: lhs.context)
  }
  
  /// Performs element-wise addition between two tensors with automatic broadcasting support.
  static func +(lhs: Tensor, rhs: Tensor) -> Tensor {
    if let axis = Tensor.axisToApplyAlong(selfSize: lhs.size,
                                          size: rhs.size) {
      return lhs.addAlong(axis: axis, value: rhs)
    }
    
    // Accelerate-backed flat element-wise add
    let result = NumSwiftFlat.add(lhs.storage, rhs.storage)
    
    let context = TensorContext { inputs, gradient, wrt in
      let copy = gradient.copy()
      copy.label = "addition_input_grad"
      return (copy, Tensor(), Tensor())
    }
    
    let new = Tensor(result, size: lhs.size, context: context)
    new.label = "addition"
    
    new.setGraphSafe(lhs)
    new.setGraphSafe(rhs)
    
    return new
  }
  
  /// Performs element-wise subtraction between two tensors with automatic broadcasting support.
  static func -(lhs: Tensor, rhs: Tensor) -> Tensor {
    if let axis = Tensor.axisToApplyAlong(selfSize: lhs.size,
                                          size: rhs.size) {
      return lhs.subtractAlong(axis: axis, value: rhs)
    }
    
    // Accelerate-backed flat element-wise subtract
    let result = NumSwiftFlat.subtract(lhs.storage, rhs.storage)
    
    let context = TensorContext { inputs, gradient, wrt in
      if let wrt, (rhs.graphChain.contains(wrt.id) || rhs.id == wrt.id) {
        return (gradient * -1, Tensor(), Tensor())
      }

      return (gradient, Tensor(), Tensor())
    }
    
    let new = Tensor(result, size: lhs.size, context: context)
    new.label = "subtraction"

    new.setGraphSafe(lhs)
    new.setGraphSafe(rhs)
    
    return new
  }
  
  /// Performs element-wise multiplication between two tensors with automatic broadcasting support.
  static func *(lhs: Tensor, rhs: Tensor) -> Tensor {
    if let axis = Tensor.axisToApplyAlong(selfSize: lhs.size,
                                          size: rhs.size) {
      return lhs.multiplyAlong(axis: axis, value: rhs)
    }
    
    // Accelerate-backed flat element-wise multiply
    let result = NumSwiftFlat.multiply(lhs.storage, rhs.storage)
    
    let copied = rhs.copy()
    let context = TensorContext { inputs, gradient, wrt in
      return (gradient * copied, Tensor(), Tensor())
    }
    
    let new = Tensor(result, size: lhs.size, context: context)
    new.label = "multiplication"

    new.setGraphSafe(lhs)
    new.setGraphSafe(rhs)
    
    return new
  }
  
  /// Performs element-wise division between two tensors with automatic broadcasting support.
  static func /(lhs: Tensor, rhs: Tensor) -> Tensor {
    if let axis = Tensor.axisToApplyAlong(selfSize: lhs.size,
                                          size: rhs.size) {
      return lhs.divideAlong(axis: axis, value: rhs)
    }
    
    // Accelerate-backed flat element-wise divide
    let result = NumSwiftFlat.divide(lhs.storage, rhs.storage)
    
    let copied = rhs.copy()
    let context = TensorContext { inputs, gradient, wrt  in
      if let wrt, (rhs.graphChain.contains(wrt.id) || rhs.id == wrt.id) {
        
        let result = gradient * (-1 * (inputs / (copied * copied)))
        
        return (result, Tensor(), Tensor())
      }

      return (gradient * (1 / copied), Tensor(), Tensor())
    }
    
    let new = Tensor(result, size: lhs.size, context: context)
    new.label = "division"

    new.setGraphSafe(lhs)
    new.setGraphSafe(rhs)
    
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
    
    // Transpose swaps columns and rows per depth slice
    let newSize = TensorSize(rows: columns, columns: rows, depth: depth)
    let sliceSize = rows * columns
    
    if depth == 1 {
      // Single depth: one Accelerate call
      let result = NumSwiftFlat.transpose(storage, rows: rows, columns: columns)
      return Tensor(result, size: newSize, context: context)
    }
    
    // Multiple depth slices: transpose each separately
    var result = Tensor.Value(repeating: 0, count: storage.count)
    for d in 0..<depth {
      let start = d * sliceSize
      let slice = Tensor.Value(storage[start..<(start + sliceSize)])
      let transposed = NumSwiftFlat.transpose(slice, rows: rows, columns: columns)
      for i in 0..<sliceSize {
        result[start + i] = transposed[i]
      }
    }
    
    return Tensor(result, size: newSize, context: context)
  }
}

// NOTE: debugDescription, Array<Tensor> extensions, fillRandom/fillWith, and Gradient operators
// have been moved to Tensor.swift as part of the flat storage refactor.

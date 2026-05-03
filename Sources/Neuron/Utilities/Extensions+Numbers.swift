//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/26/22.
//

import Foundation

public extension Int {
  /// Converts the integer to a `Tensor.Scalar` value (`Float` or `Float16` depending on build configuration).
  var asTensorScalar: Tensor.Scalar {
    Tensor.Scalar(self)
  }
}

public extension Array where Element == [[Tensor.Scalar]] {
  /// Extracts a 3D sub-array using separate column, row, and depth range expressions.
  ///
  /// - Parameters:
  ///   - colRange: Range of column indices to extract.
  ///   - rowRange: Range of row indices to extract.
  ///   - depthRange: Range of depth indices to extract.
  /// - Returns: A new 3D array containing the selected elements.
  subscript(_ colRange: some RangeExpression<Int>,
            _ rowRange: some RangeExpression<Int>,
            _ depthRange: some RangeExpression<Int>) -> [[[Tensor.Scalar]]] {
    var data: [[[Tensor.Scalar]]] = []
    
    for d in depthRange.relative(to: self) {
      var rows: [[Tensor.Scalar]] = []
      for r in rowRange.relative(to: self[d]) {
        var row: [Tensor.Scalar] = []
        for c in colRange.relative(to: self[d][r]) {
          row.append(self[d][r][c])
        }
        rows.append(row)
      }
      data.append(rows)
    }
    
    return data
  }
}

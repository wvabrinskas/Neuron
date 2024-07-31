//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/26/22.
//

import Foundation

public extension Int {
  func asTensorScalar<N: TensorNumeric>() -> Tensor<N>.Scalar {
    Tensor<N>.Scalar(self)

  }
}

public extension Array where Element == [[Tensor<Float16>.Scalar]] {
  subscript(_ colRange: some RangeExpression<Int>,
                              _ rowRange: some RangeExpression<Int>,
                              _ depthRange: some RangeExpression<Int>) -> [[[Tensor<Float16>.Scalar]]] {
    var data: [[[Tensor<Float16>.Scalar]]] = []
    
    for d in depthRange.relative(to: self) {
      var rows: [[Tensor<Float16>.Scalar]] = []
      for r in rowRange.relative(to: self[d]) {
        var row: [Tensor<Float16>.Scalar] = []
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

public extension Array where Element == [[Tensor<Float>.Scalar]] {
  subscript(_ colRange: some RangeExpression<Int>,
                              _ rowRange: some RangeExpression<Int>,
                              _ depthRange: some RangeExpression<Int>) -> [[[Tensor<Float>.Scalar]]] {
    var data: [[[Tensor<Float>.Scalar]]] = []
    
    for d in depthRange.relative(to: self) {
      var rows: [[Tensor<Float>.Scalar]] = []
      for r in rowRange.relative(to: self[d]) {
        var row: [Tensor<Float>.Scalar] = []
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

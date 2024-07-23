//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/26/22.
//

import Foundation

public extension Int {
  var asTensorScalar: Tensor.Scalar {
    Tensor.Scalar(self)
  }
}

public extension Array where Element == [[Tensor.Scalar]] {
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

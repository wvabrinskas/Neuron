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

public extension Array where Element == [[Float]] {
  subscript(_ colRange: some RangeExpression<Int>,
            _ rowRange: some RangeExpression<Int>,
            _ depthRange: some RangeExpression<Int>) -> [[[Float]]] {
    var data: [[[Float]]] = []
    
    for d in depthRange.relative(to: self) {
      var rows: [[Float]] = []
      for r in rowRange.relative(to: self[d]) {
        var row: [Float] = []
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

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/28/22.
//

import Foundation

/// Object that defines a size of a Tensor
public struct TensorSize: Codable, Equatable {
  public let rows: Int
  public let columns: Int
  public let depth: Int
  public var isEmpty: Bool {
    return rows == 0 && columns == 0 && depth == 0
  }
  public var asArray: [Int] {
    [columns, rows, depth]
  }
  
  /// Initializer that takes in array of any size. Will take the first three elements and construct row, columns, depth. `Rows` is from index 1, `columns` is index 0, and `depth` is index 2.
  /// - Parameter array: Array to build the tensor size
  public init(array: [Int]) {
    rows = array[safe: 1, 0]
    columns = array[safe: 0, 0]
    depth = array[safe: 2, 0]
  }
  
  /// Default initializer
  /// - Parameters:
  ///   - rows: Number that defines the row count
  ///   - columns: Number that defines the column count
  ///   - depth: Number that defines the depth count
  public init(rows: Int = 0, columns: Int = 0, depth: Int = 0) {
    self.rows = rows
    self.columns = columns
    self.depth = depth
  }
}

public extension Array where Element == Int {
  var tensorSize: TensorSize {
    return TensorSize(array: self)
  }
}

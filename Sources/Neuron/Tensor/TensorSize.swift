//
//  File.swift
//  
//
//  Created by William Vabrinskas on 7/28/22.
//

import Foundation

/// Object that defines a size of a Tensor
public struct TensorSize: Codable, Equatable {
  /// The number of rows in the tensor.
  public let rows: Int
  /// The number of columns in the tensor.
  public let columns: Int
  /// The depth (third dimension) of the tensor.
  public let depth: Int
  /// Indicates whether the tensor size has no extent, i.e., all dimensions are zero.
  /// - Returns: `true` if rows, columns, and depth are all zero; otherwise `false`.
  public var isEmpty: Bool {
    return rows == 0 && columns == 0 && depth == 0
  }
  /// Returns the tensor dimensions as an array in `[columns, rows, depth]` order.
  /// - Returns: An array of three integers representing `[columns, rows, depth]`.
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

extension TensorSize: CustomDebugStringConvertible {
  /// A human-readable description of the tensor size in `(columns, rows, depth)` format.
  /// - Returns: A string describing the tensor size.
  public var debugDescription: String {
    "TensorSize(\(columns), \(rows), \(depth))"
  }
}

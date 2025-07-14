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
  public private(set) var features: Int = 1
  public var isEmpty: Bool {
    return rows == 0 && columns == 0 && depth == 0
  }
  public var asArray: [Int] {
    [columns, rows, depth]
  }
  
  /// Initializer that takes in array of any size. Will take the first three elements and construct row, columns, depth. `Rows` is from index 1, `columns` is index 0, and `depth` is index 2.
  /// - Parameter array: Array to build the tensor size
  public init(array: [Int]) {
    columns = array[safe: 0, 0]
    rows = array[safe: 1, 0]
    depth = array[safe: 2, 0]
    
    setFeatures()
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
    
    setFeatures()
  }
  
  // this doesnt work for a 3D array as 32x32x1 has 1 feature not 32
  // 3D -> 32x32x1 -> assume rows & columns are the same size, since that should be enforceable in other layers too? 
  // 2D -> 32x12x1 ->
  // 1D -> 32x1x1
  // this is not great... we can't assume square inputs
  mutating private func setFeatures() {
    if depth > 1 || (depth == 1 && rows == columns) {
      features = depth
    } else if rows > 1 {
      features = rows
    } else {
      features = columns
    }
  }
}

public extension Array where Element == Int {
  var tensorSize: TensorSize {
    return TensorSize(array: self)
  }
}

extension TensorSize: CustomDebugStringConvertible {
  public var debugDescription: String {
    "TensorSize(\(columns), \(rows), \(depth)) features: \(features)"
  }
}

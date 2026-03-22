//
//  File.swift
//
//
//  Created by William Vabrinskas on 7/28/22.
//

import Foundation

/// Object that defines a size of a Tensor
public struct TensorSize: Codable, Equatable, Comparable {

  /// The number of rows in the tensor.
  public let rows: Int
  /// The number of columns in the tensor.
  public let columns: Int
  /// The depth (third dimension) of the tensor.
  public let depth: Int
  
  /// The number of batches in the tensor.
  public let batchCount: Int
  
  /// Indicates whether the tensor size has no extent, i.e., all dimensions are zero.
  /// - Returns: `true` if rows, columns, and depth are all zero; otherwise `false`.
  public var isEmpty: Bool {
    return rows == 0 && columns == 0 && depth == 0 && batchCount == 0
  }
  /// Returns the tensor dimensions as an array in `[columns, rows, depth]` order. If there is more than 1 in the batch it will be added to the end fo the array.
  /// - Returns: An array of three integers representing `[columns, rows, depth]`.
  public var asArray: [Int] {
    [columns, rows, depth, batchCount.nilIfOne].compactMap { $0 } // don't need to return batchCount if it's one
  }
  
  /// Returns the tensor dimensions as an array in `[columns, rows, depth]` order, excluding the batch dimension.
  /// - Returns: An array of three integers representing `[columns, rows, depth]`.
  public var unitSize: [Int] {
    [columns, rows, depth]
  }
  
  /// Coding keys used for encoding and decoding a `TensorSize` instance.
  public enum CodingKeys: String, CodingKey {
    case rows, columns, depth, batchCount
  }
  
  /// Returns a Boolean value indicating whether the total element count of the left tensor is less than that of the right tensor.
  /// - Parameter lhs: The left-hand side `TensorSize` to compare.
  /// - Parameter rhs: The right-hand side `TensorSize` to compare.
  /// - Returns: `true` if the total number of elements in `lhs` is less than in `rhs`; otherwise, `false`.
  public static func < (lhs: TensorSize, rhs: TensorSize) -> Bool {
    (lhs.columns * lhs.rows * lhs.depth) < (rhs.columns * rhs.rows * rhs.depth)
  }
  
  
  /// Returns a Boolean value indicating whether the total element count of the left tensor is greater than that of the right tensor.
  /// - Parameter lhs: The left-hand side `TensorSize` to compare.
  /// - Parameter rhs: The right-hand side `TensorSize` to compare.
  /// - Returns: `true` if the total number of elements in `lhs` is greater than in `rhs`; otherwise, `false`.
  public static func > (lhs: TensorSize, rhs: TensorSize) -> Bool {
    (lhs.columns * lhs.rows * lhs.depth) > (rhs.columns * rhs.rows * rhs.depth)
  }
  
  
  /// Creates a `TensorSize` by decoding from the given decoder.
  /// - Parameter decoder: The decoder to read data from.
  /// - Throws: An error if any required value is missing or cannot be decoded.
  public init(from decoder: any Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    self.rows = try container.decode(Int.self, forKey: .rows)
    self.columns = try container.decode(Int.self, forKey: .columns)
    self.depth = try container.decode(Int.self, forKey: .depth)
    self.batchCount = try container.decodeIfPresent(Int.self, forKey: .batchCount) ?? 1
  }
  
  /// Encodes the `TensorSize` into the given encoder.
  /// - Parameter encoder: The encoder to write data to.
  /// - Throws: An error if any value cannot be encoded.
  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(rows, forKey: .rows)
    try container.encode(columns, forKey: .columns)
    try container.encode(depth, forKey: .depth)
    try container.encodeIfPresent(batchCount, forKey: .batchCount)
  }
  
  /// Initializer that takes in array of any size. Will take the first three elements and construct row, columns, depth. `Rows` is from index 1, `columns` is index 0, and `depth` is index 2.
  /// - Parameter array: Array to build the tensor size
  public init(array: [Int]) {
    rows = array[safe: 1, 0]
    columns = array[safe: 0, 0]
    depth = array[safe: 2, 0]
    batchCount = array[safe: 3, 1]
  }
  
  /// Default initializer
  /// - Parameters:
  ///   - rows: Number that defines the row count
  ///   - columns: Number that defines the column count
  ///   - depth: Number that defines the depth count
  public init(rows: Int = 0, columns: Int = 0, depth: Int = 0, batchCount: Int = 1) {
    self.rows = rows
    self.columns = columns
    self.depth = depth
    self.batchCount = batchCount
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
    """
    TensorSize 
    
    rows: \(rows)
    columns: \(columns)
    depth: \(depth)
    batchCount: \(batchCount)
    
    """
  }
}

extension Int {
  var nilIfZero: Int? {
    self == 0 ? nil : self
  }
  
  var nilIfOne: Int? {
    self == 1 ? nil : self
  }
}

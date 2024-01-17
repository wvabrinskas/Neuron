//
//  File.swift
//  
//
//  Created by William Vabrinskas on 6/1/23.
//

import Foundation
import NumSwiftC
import NumSwift

public typealias VectorizableItem = Hashable & Equatable

public enum VectorFormat {
  case start, end, none
}

public protocol Vectorizing {
  associatedtype Item: VectorizableItem
  typealias Vector = [Item: Int]
  typealias InverseVector = [Int: Item]
  /// Value that indicate starting of a vactor
  var start: Int { get }
  /// Value that indicate ending of a vactor
  var end: Int { get }
  /// The current full vector storage of every value passed in keyed by the `Item`
  var vector: Vector { get }
  /// The current full vector storage of every value passed in keyed by the `Index`
  var inverseVector: InverseVector { get }
  @discardableResult
  func vectorize(_ items: [Item], format: VectorFormat) -> [Int]
  func unvectorize(_ vector: [Int]) -> [Item]
  func unvectorizeOneHot(_ vector: Tensor) -> [Item]
  func oneHot(_ items: [Item]) -> Tensor
}


/// Takes an input and turns it in to a vector array of integers indicating its value.
/// ex. Can take a string and apply a integer value to the word so that if it came up again
/// it would return the same integer value for that word.
public class Vectorizer<T: VectorizableItem>: Vectorizing {
  public typealias Item = T
  public private(set) var vector: Vector = [:]
  public private(set) var inverseVector: InverseVector = [:]
  
  /// Value that indicate starting of a vactor
  public let start: Int = 0
  /// Value that indicate ending of a vactor
  public let end: Int = 1

  /// max index is the first index we can use to identify words
  /// this means we reserve the start and end indicies labels
  private var maxIndex: Int = 2
  
  private var maxItemLength: Int = 0
  
  public init() {}
  
  
  /// One hot vectorizes a input that has already been vectorized.
  ///  `NOTE: Please call `vectorize` on your input first before calling `oneHot` otherwise it will not work
  /// - Parameter items: Array or `Item` to oneHot encode
  /// - Returns: The encoded one hot vector as a 3D tensor where the depth is the length of `items`.
  public func oneHot(_ items: [T]) -> Tensor {
    var result: Tensor.Data = []
    
    for i in 0..<items.count {
      var vectorized: [Tensor.Scalar] = [Float](repeating: 0, count: maxIndex - 2)
      
      let item = formatItem(item: items[i])
      
      if let inVector = vector[item] {
        let adjustedIndex = max(0, inVector - 2) // offset by 2 since we saved the first two indexes for start and end labels
        
        if adjustedIndex < vectorized.count {
          vectorized[adjustedIndex] = 1.asTensorScalar
        }
      }
      
      result.append([vectorized])
    }
      
    return Tensor(result)
  }
  
  @discardableResult
  public func vectorize(_ items: [T], format: VectorFormat = .none) -> [Int] {
    var vectorized: [Int] = []
    
    if format == .start {
      vectorized = [start]
    }
    
    var lastKey = maxIndex

    items.forEach { item in
      let key = formatItem(item: item)
      if vector[key] == nil {
        vector[key] = lastKey
        inverseVector[lastKey] = key
        vectorized.append(lastKey)
        lastKey += 1
      } else if let itemVector = vector[key] {
        vectorized.append(itemVector)
      }
    }
    
    maxIndex = lastKey
    
    if format == .end {
      vectorized.append(end)
    }

    return vectorized
  }
  
  public func unvectorizeOneHot(_ vector: Tensor) -> [T] {
    var items: [T] = []
    
    vector.value.forEach { v in
      if let indexOfHot = v[0].firstIndex(of: 1) {
        if let s = inverseVector[Int(indexOfHot + 2)] {
          items.append(s)
        }
      }
    }
    
    return items
  }
  
  
  public func unvectorize(_ vector: [Int]) -> [T] {
    var items: [T] = []
    
    vector.forEach { v in
      if let i = inverseVector[v] {
        items.append(i)
      }
    }
    
    return items
  }
  
  // MARK: Private
  func formatItem(item: T) -> T {
    if let i = item as? String {
      return i.lowercased() as? T ?? item
    }
    
    return item
  }
}

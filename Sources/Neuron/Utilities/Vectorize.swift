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

public protocol Vectorizing {
  associatedtype Item: VectorizableItem
  typealias Vector = [Item: Int]
  var vector: Vector { get }
  func vectorize(_ items: [Item]) -> [Int]
}


public class Vectorizer<T: VectorizableItem>: Vectorizing {
  public typealias Item = T
  public private(set) var vector: Vector = [:]
  
  private var maxIndex: Int = 0
  
  public func vectorize(_ items: [T]) -> [Int] {
    var vectorized: [Int] = []
    var lastKey = maxIndex

    items.forEach { item in
      let key = formatItem(item: item)
      if vector[key] == nil {
        vector[key] = lastKey + 1
        lastKey += 1
        vectorized.append(lastKey)
      } else if let itemVector = vector[key] {
        vectorized.append(itemVector)
      }
    }
    
    maxIndex = lastKey
    
    return vectorized
  }
  
  // MARK: Private
  func formatItem(item: T) -> T {
    if let i = item as? String {
      return i.lowercased() as? T ?? item
    }
    
    return item
  }
}

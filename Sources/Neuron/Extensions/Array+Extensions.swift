//
//  Array+Extensions.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/23/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public extension Array where Element: Equatable & Numeric & FloatingPoint {
  static func /(lhs: [Element], rhs: [Element]) -> [Element] {
    precondition(lhs.count == rhs.count)
    
    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = lhs[i]
      let right = rhs[i]
      addedArray.append(left / right)
    }
    
    return addedArray
  }
  
  static func /(lhs: [Element], rhs: Element) -> [Element] {
    return lhs.map({ $0 / rhs })
  }
}

public extension Array where Element: Equatable & Numeric {
  func add(add: [Element]) -> [Element] {
    precondition(self.count == add.count)

    var addedArray: [Element] = []
    
    for i in 0..<self.count {
      let left = self[i]
      let right = add[i]
      addedArray.append(left + right)
    }
      
    return addedArray
  }
  
  static func +=(lhs: inout [Element], rhs: [Element]) {
    precondition(lhs.count == rhs.count)

    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = rhs[i]
      let right = lhs[i]
      addedArray.append(left + right)
    }
      
    lhs = addedArray
  }
  
  static func +(lhs: [Element], rhs: [Element]) -> [Element] {
    precondition(lhs.count == rhs.count)

    var addedArray: [Element] = []
    
    for i in 0..<rhs.count {
      let left = lhs[i]
      let right = rhs[i]
      addedArray.append(left + right)
    }
    
    return addedArray
  }
}

public extension Array where Element: Equatable {
  
  func batched(into size: Int) -> [[Element]] {
    return stride(from: 0, to: count, by: size).map {
      Array(self[$0 ..< Swift.min($0 + size, count)])
    }
  }
  /// Get a copy of self but with randomized data indexes
  /// - Returns: Returns Self but with the data randomized
  func randomize() -> Self {
    var arrayCopy = self
    var randomArray: [Element] = []
    
    for _ in 0..<self.count {
      guard let element = arrayCopy.randomElement() else {
        break
      }
      randomArray.append(element)
      
      if let index = arrayCopy.firstIndex(of: element) {
        arrayCopy.remove(at: index)
      }
    }
    
    return randomArray
  }

}

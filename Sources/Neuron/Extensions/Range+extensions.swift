//
//  Range+extensions.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/21/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public extension Range where Element: Strideable {
  
  /// Get Range object as array
  /// - Returns: Returns every element in the range in order as an array of its Elements
  func array() -> [Element] {
    var newArray: [Element] = []
    
    for i in self {
      newArray.append(i)
    }
    return newArray
  }
}


@propertyWrapper
public struct TestNaN {
  public var wrappedValue: Float {
      didSet {
        if wrappedValue.isNaN {
          fatalError("number is broken =( ")
        }
      }
  }

  public init(wrappedValue: Float) {
      self.wrappedValue = wrappedValue
  }
}

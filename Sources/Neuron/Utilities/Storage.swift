//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/30/24.
//

import Foundation

public final class ThreadStorage<T> {
  @Atomic
  public var storage: [Int: T] = [:]
  
  public func store(_ value: T, at index: Int) {
    storage[index] = value
  }
  
  public func value(at: Int) -> T {
    guard let val = storage[at] else {
      fatalError("could not find object for thread id: \(at)")
    }
    
    return val
  }
}



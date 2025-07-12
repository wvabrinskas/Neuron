//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/30/24.
//

import Foundation

public final class ThreadStorage<T> {
  private(set) var storage: [Int: T] = [:]
  private let lock = NSLock()
  
  public func store(_ value: T, at index: Int) {
    lock.with {
      storage[index] = value
    }
  }
  
  public func clear() {
    lock.with {
      storage.removeAll()
    }
  }
  
  public func value(at: Int) -> T? {
    defer {
      lock.unlock()
    }
    lock.lock()
    guard let val = storage[at] else {
      return nil
    }
    
    return val
  }
  
  public subscript (index: Int) -> T? {
    get {
      return value(at: index)
    } set {
      guard let newValue else {
        return
      }
      store(newValue, at: index)
    }
  }
}



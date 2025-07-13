//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/30/24.
//

import Foundation

public final class ThreadStorage<Key: Hashable, T> {
  private(set) var storage: [Key: T?] = [:]
  private let lock = NSRecursiveLock()
  private var inSync = false
  
  var defaultValue: T
  
  init(defaultValue: T) {
    self.defaultValue = defaultValue
  }
  
  public func store(_ value: T?, at key: Key) {
    lock.with {
      inSync = false
      storage[key] = value
    }
  }
  
  public func clear() {
    lock.with {
      inSync = false
      storage.removeAll()
    }
  }
  
  public func value(at: Key) -> T? {
    defer {
      lock.unlock()
    }
    lock.lock()
    guard let val = storage[at] else {
      return defaultValue
    }
    
    return val
  }
  
  public func sync(_ value: T) {
    defer {
      lock.unlock()
    }
    lock.lock()
    guard inSync == false else { return }
    storage.removeAll()
    defaultValue = value
    inSync = true
  }
  
  public subscript (index: Key) -> T? {
    get {
      return value(at: index)
    } set {
      store(newValue, at: index)
    }
  }
}


extension ThreadStorage where T == [Tensor.Scalar] {
  public func reduceMean() -> T {
    defer {
      lock.unlock()
    }
    
    lock.lock()
    let nonNilValues = storage.values.compactMap((\.self))
    let average = Tensor(nonNilValues).mean(axis: 0)
    return average.value[safe: 0]?[safe: 0] ?? []
  }
}

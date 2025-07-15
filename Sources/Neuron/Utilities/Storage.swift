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
  
  var defaultValue: T?
  
  init(defaultValue: T?) {
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

extension ThreadStorage where T == Tensor.Scalar {
  public func reduceMean(batchSize: T, features: T) -> T {
    defer {
      lock.unlock()
    }
    
    lock.lock()
    let nonNilValues = storage.values.compactMap((\.self))
    let average = Tensor(nonNilValues).sum(axis: -1) / (batchSize)
    return average.asScalar()
  }
  
  public func reduceVariance(mean: T, batchSize: T, features: T) -> T {
    defer {
      lock.unlock()
    }
    
    lock.lock()
    let nonNilValues = storage.values.compactMap((\.self))
    // multiply by features here because this array with only contain the sums
    let variance = (Tensor(nonNilValues).sum(axis: -1) - Tensor.Scalar.pow(mean, 2)) / (batchSize)
    return variance.asScalar()
  }
}

extension ThreadStorage where T == [[Tensor.Scalar]] {
  public func welfordVariance(batchSize: Tensor.Scalar, features: Tensor.Scalar) -> T {
    let sum = Tensor(storage.values.compactMap { $0 }).sum(axis: 2) / batchSize
    
    return sum.value[safe: 0] ?? []
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

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/3/22.
//

import Foundation

public extension Sequence {
  func asyncMap<T>(
    _ transform: (Element) async throws -> T
  ) async rethrows -> [T] {
    var values = [T]()
    
    for element in self {
      try await values.append(transform(element))
    }
    
    return values
  }
}

public extension Collection {
  func prettyPrint() {
    var currentElement: Any = self
    
    while let current = currentElement as? Array<Any> {
      if let next = current.first {
        currentElement = next
      } else {
        break
      }
    }
  }
}

public extension NSLock {
  @discardableResult
  func with<T>(_ block: () throws -> T) rethrows -> T {
    lock()
    defer { unlock() }
    return try block()
  }
}

@propertyWrapper
public struct Atomic<Value> {
  private let lock = NSLock()
  private var value: Value
  
  public init(wrappedValue: Value) {
    self.value = wrappedValue
  }
  
  public var wrappedValue: Value {
    get {
      lock.lock()
      defer { lock.unlock() }
      return value
    }
    set {
      lock.lock()
      value = newValue
      lock.unlock()
    }
  }
}

extension OperationQueue {
  
  func addSynchronousOperation(barrier: Bool = false, _ block: @escaping () -> ()) {
    let group = DispatchGroup()
    group.enter()
    
    if barrier {
      self.addBarrierBlock {
        block()
        group.leave()
      }
      
    } else {
      self.addOperation {
        block()
        group.leave()
      }
      
    }

    group.wait()
  }
}

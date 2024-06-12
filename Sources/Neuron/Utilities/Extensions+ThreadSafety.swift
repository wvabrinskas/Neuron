//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/3/22.
//

import Foundation

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
      defer { lock.unlock() }
      
      lock.lock()
      value = newValue
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

public final class SynchronousOperationQueue: OperationQueue, @unchecked Sendable {
  
  public init(name: String? = nil) {
    super.init()
    
    self.name = name
    maxConcurrentOperationCount = 1
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/3/22.
//

import Foundation

public extension NSRecursiveLock {
  @discardableResult
  /// Executes a closure while holding the recursive lock.
  ///
  /// - Parameter block: Work performed while locked.
  /// - Returns: Value returned from `block`.
  func with<T>(_ block: () throws -> T) rethrows -> T {
    lock()
    defer { unlock() }
    return try block()
  }
}

public extension NSLock {
  @discardableResult
  /// Executes a closure while holding the lock.
  ///
  /// - Parameter block: Work performed while locked.
  /// - Returns: Value returned from `block`.
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
  
  /// Creates an atomic wrapper around an initial value.
  ///
  /// - Parameter wrappedValue: Initial stored value.
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
  
  /// Creates a serial operation queue for synchronous barrier-style workloads.
  ///
  /// - Parameter name: Optional queue name for diagnostics.
  public init(name: String? = nil) {
    super.init()
    
    self.name = name
    maxConcurrentOperationCount = 1
  }
}

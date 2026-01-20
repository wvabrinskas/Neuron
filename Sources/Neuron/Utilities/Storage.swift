//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/30/24.
//

import Foundation

/// A thread-safe storage container that manages key-value pairs with optional values.
///
/// `AtomicStorage` provides a dictionary-like interface with built-in thread safety using locks.
/// It supports storing optional values, setting a default value for missing keys, and synchronization
/// operations that reset the storage to a new default state.
///
/// Example usage:
/// ```swift
/// let storage = AtomicStorage<String, Int>(defaultValue: 0)
/// storage.store(42, at: "answer")
/// let value = storage.value(at: "answer") // Returns 42
/// let missing = storage.value(at: "unknown") // Returns 0 (default)
/// ```
///
/// - Note: All operations are thread-safe and protected by an internal lock.
public final class AtomicStorage<Key: Hashable, T> {
  /// The internal dictionary that holds the stored key-value pairs.
  ///
  /// Values are stored as optionals to distinguish between explicitly stored `nil` values
  /// and missing keys.
  private(set) var storage: [Key: T?] = [:]
  
  /// The lock used to ensure thread-safe access to the storage.
  private let lock = NSLock()
  
  /// Indicates whether the storage is currently synchronized.
  ///
  /// When `true`, the storage has been reset to a synchronized state with a new default value.
  /// This flag is set to `false` whenever new values are stored or the storage is cleared.
  private var inSync = false
  
  /// The default value returned when accessing a key that doesn't exist in storage.
  var defaultValue: T?
  
  /// Creates a new atomic storage instance with the specified default value.
  ///
  /// - Parameter defaultValue: The value to return when accessing keys that haven't been stored.
  ///   Can be `nil` if no default behavior is desired.
  init(defaultValue: T?) {
    self.defaultValue = defaultValue
  }
  
  /// Stores a value at the specified key in a thread-safe manner.
  ///
  /// This operation marks the storage as not synchronized and acquires a lock to ensure
  /// thread safety during the write operation.
  ///
  /// - Parameters:
  ///   - value: The value to store. Can be `nil` to explicitly store a nil value.
  ///   - key: The key at which to store the value.
  public func store(_ value: T?, at key: Key) {
    lock.with {
      inSync = false
      storage[key] = value
    }
  }
  
  /// Removes all values from storage in a thread-safe manner.
  ///
  /// This operation marks the storage as not synchronized but preserves the default value.
  public func clear() {
    lock.with {
      inSync = false
      storage.removeAll()
    }
  }
  
  /// Retrieves the value associated with the specified key.
  ///
  /// If the key exists in storage, returns its associated value (which may be `nil`).
  /// If the key doesn't exist, returns the `defaultValue`.
  ///
  /// - Parameter at: The key whose associated value should be retrieved.
  /// - Returns: The value associated with the key, or the default value if the key doesn't exist.
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
  
  /// Synchronizes the storage to a new default value, clearing all existing entries.
  ///
  /// This operation removes all stored key-value pairs and sets a new default value.
  /// If the storage is already synchronized, this operation has no effect.
  ///
  /// - Parameter value: The new default value to use for the synchronized storage.
  ///
  /// - Note: This is useful for resetting the storage to a known state, such as when
  ///   reinitializing after a configuration change.
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
  
  /// Accesses the value associated with the given key for reading and writing.
  ///
  /// Use subscript syntax to access and modify values in the storage:
  ///
  /// ```swift
  /// storage["myKey"] = 100  // Store a value
  /// let value = storage["myKey"]  // Retrieve a value
  /// ```
  ///
  /// - Parameter index: The key to use for accessing the storage.
  /// - Returns: The value associated with the key, or the default value if the key doesn't exist.
  public subscript (index: Key) -> T? {
    get {
      return value(at: index)
    } set {
      store(newValue, at: index)
    }
  }
}

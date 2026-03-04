//
//  TensorStorage.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/4/26.
//

import Foundation

/// A pointer-backed contiguous memory container for tensor data.
///
/// `TensorStorage` manages a block of `Scalar` values via raw pointers,
/// providing subscript access, Collection conformance, and Codable support.
/// It uses reference semantics (class) so that subclasses (e.g. `MetalTensorStorage`)
/// can back the same pointer interface with `MTLBuffer` memory.
///
/// Memory lifecycle: the base class allocates via `UnsafeMutablePointer.allocate(capacity:)`
/// and deallocates in `deinit`. Subclasses that provide externally-managed memory
/// should use the `init(pointer:count:deallocator:)` initializer with a custom deallocator.
public final class TensorStorage {
  #if QUANTIZED_F16
  public typealias Scalar = Float16
  #else
  public typealias Scalar = Float
  #endif

  /// Raw pointer to the contiguous storage.
  public private(set) var pointer: UnsafeMutablePointer<Scalar>

  /// Number of scalar elements in this storage.
  public let count: Int

  /// Closure called on deallocation. Nil means this instance owns and will deallocate the pointer.
  private let deallocator: (() -> Void)?

  // MARK: - Initializers

  /// Allocates uninitialized storage for `count` elements (zeroed).
  public init(count: Int) {
    self.count = count
    if count > 0 {
      self.pointer = .allocate(capacity: count)
      self.pointer.initialize(repeating: 0, count: count)
    } else {
      self.pointer = .allocate(capacity: 1)
    }
    self.deallocator = nil
  }

  /// Allocates storage filled with `repeating` value.
  public init(repeating value: Scalar, count: Int) {
    self.count = count
    if count > 0 {
      self.pointer = .allocate(capacity: count)
      self.pointer.initialize(repeating: value, count: count)
    } else {
      self.pointer = .allocate(capacity: 1)
    }
    self.deallocator = nil
  }

  /// Allocates storage and copies data from an `Array`.
  public init(_ array: [Scalar]) {
    let n = array.count
    self.count = n
    self.deallocator = nil
    if n == 0 {
      self.pointer = .allocate(capacity: 1)
    } else {
      let ptr = UnsafeMutablePointer<Scalar>.allocate(capacity: n)
      array.withUnsafeBufferPointer { src in
        ptr.initialize(from: src.baseAddress!, count: n)
      }
      self.pointer = ptr
    }
  }

  /// Allocates storage and copies data from a `ContiguousArray`.
  public init(_ contiguous: ContiguousArray<Scalar>) {
    let n = contiguous.count
    self.count = n
    self.deallocator = nil
    if n == 0 {
      self.pointer = .allocate(capacity: 1)
    } else {
      let ptr = UnsafeMutablePointer<Scalar>.allocate(capacity: n)
      contiguous.withUnsafeBufferPointer { src in
        ptr.initialize(from: src.baseAddress!, count: n)
      }
      self.pointer = ptr
    }
  }

  /// Subclass / external-memory initializer.
  /// The caller provides a pointer whose lifetime is managed by `deallocator`.
  /// Pass an empty deallocator if the pointer's lifetime is managed elsewhere
  /// (e.g. by an `MTLBuffer` retained by the subclass).
  public init(pointer: UnsafeMutablePointer<Scalar>, count: Int, deallocator: @escaping () -> Void) {
    self.pointer = pointer
    self.count = count
    self.deallocator = deallocator
  }

  deinit {
    if let deallocator {
      deallocator()
    } else {
      if count > 0 {
        pointer.deinitialize(count: count)
      } else {
        pointer.deinitialize(count: 1)
      }
      pointer.deallocate()
    }
  }

  // MARK: - Subscript

  public subscript(index: Int) -> Scalar {
    get {
      assert(index >= 0 && index < count, "TensorStorage index \(index) out of range [0..<\(count)]")
      return pointer[index]
    }
    set {
      assert(index >= 0 && index < count, "TensorStorage index \(index) out of range [0..<\(count)]")
      pointer[index] = newValue
    }
  }

  // MARK: - Properties

  public var isEmpty: Bool { count == 0 }

  // MARK: - Bridging

  /// Copies data out to a `ContiguousArray<Scalar>`.
  public func toContiguousArray() -> ContiguousArray<Scalar> {
    guard count > 0 else { return ContiguousArray<Scalar>() }
    return ContiguousArray<Scalar>(UnsafeBufferPointer(start: pointer, count: count))
  }

  /// Copies data out to an `[Scalar]`.
  public func toArray() -> [Scalar] {
    guard count > 0 else { return [] }
    return Array(UnsafeBufferPointer(start: pointer, count: count))
  }

  // MARK: - Unsafe Access

  /// Calls `body` with an `UnsafeBufferPointer` covering the entire storage.
  @discardableResult
  public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) throws -> R) rethrows -> R {
    try body(UnsafeBufferPointer(start: pointer, count: count))
  }

  /// Calls `body` with an `UnsafeMutableBufferPointer` covering the entire storage.
  @discardableResult
  public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R) rethrows -> R {
    try body(UnsafeMutableBufferPointer(start: pointer, count: count))
  }

  // MARK: - Copy

  /// Creates a new `TensorStorage` with independently allocated memory containing the same data.
  public func copy() -> TensorStorage {
    let new = TensorStorage(count: count)
    if count > 0 {
      new.pointer.update(from: pointer, count: count)
    }
    return new
  }
}

// MARK: - Equatable

extension TensorStorage: Equatable {
  public static func == (lhs: TensorStorage, rhs: TensorStorage) -> Bool {
    guard lhs.count == rhs.count else { return false }
    guard lhs.count > 0 else { return true }
    return memcmp(lhs.pointer, rhs.pointer, lhs.count * MemoryLayout<Scalar>.stride) == 0
  }
}

// MARK: - Sequence & Collection

extension TensorStorage: Sequence {
  public func makeIterator() -> TensorStorageIterator {
    TensorStorageIterator(storage: self)
  }

  public struct TensorStorageIterator: IteratorProtocol {
    private let storage: TensorStorage
    private var index: Int = 0

    init(storage: TensorStorage) {
      self.storage = storage
    }

    public mutating func next() -> Scalar? {
      guard index < storage.count else { return nil }
      let value = storage.pointer[index]
      index += 1
      return value
    }
  }
}

extension TensorStorage: RandomAccessCollection, MutableCollection {
  public var startIndex: Int { 0 }
  public var endIndex: Int { count }

  public func index(after i: Int) -> Int { i + 1 }
  public func index(before i: Int) -> Int { i - 1 }
}

// MARK: - Codable

extension TensorStorage: Codable {
  public convenience init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    let array = try container.decode([Scalar].self)
    self.init(array)
  }

  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(toArray())
  }
}

// MARK: - CustomDebugStringConvertible

extension TensorStorage: CustomDebugStringConvertible {
  public var debugDescription: String {
    let preview = Swift.min(count, 8)
    let elements = (0..<preview).map { String(describing: pointer[$0]) }
    let suffix = count > preview ? ", ... (\(count) total)" : ""
    return "TensorStorage[\(count)](\(elements.joined(separator: ", "))\(suffix))"
  }
}

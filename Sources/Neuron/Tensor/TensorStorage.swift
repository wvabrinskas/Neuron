//
//  TensorStorage.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/4/26.
//

import Foundation

/// A pointer-backed contiguous memory container for tensor data with copy-on-write semantics.
///
/// `TensorStorage` manages a block of `Tensor.Scalar` values via raw pointers,
/// providing subscript access, Collection conformance, and Codable support.
/// Multiple `TensorStorage` instances can share the same underlying memory;
/// the actual copy is deferred until the first mutation (copy-on-write).
///
/// Memory lifecycle: the base class allocates via `UnsafeMutablePointer.allocate(capacity:)`
/// and deallocates when the last reference to the inner buffer is released.
/// Subclasses that provide externally-managed memory should use the
/// `init(pointer:count:deallocator:)` initializer with a custom deallocator.
public class TensorStorage {
  // MARK: - Inner Buffer (reference-counted memory owner)
  /// A type alias for a mutable pointer to the tensor's scalar elements.

  /// A reference-counted buffer managing a contiguous block of scalar memory.
  ///
  /// Allocates and zero-initializes storage for `count` scalars on creation.
  /// Supports custom deallocation via an optional deallocator closure.
  /// - Parameter count: The number of scalar elements to allocate storage for.
  public typealias Pointer = UnsafeMutablePointer<Tensor.Scalar>

  final class Buffer {
    var pointer: Pointer
    let count: Int
    private let deallocator: (() -> Void)?

    init(count: Int) {
      self.count = count
      self.deallocator = nil
      if count > 0 {
        self.pointer = .allocate(capacity: count)
        self.pointer.initialize(repeating: 0, count: count)
      } else {
        self.pointer = .allocate(capacity: 1)
      }
    }

    init(repeating value: Tensor.Scalar, count: Int) {
      self.count = count
      self.deallocator = nil
      if count > 0 {
        self.pointer = .allocate(capacity: count)
        self.pointer.initialize(repeating: value, count: count)
      } else {
        self.pointer = .allocate(capacity: 1)
      }
    }

    init(pointer: Pointer, count: Int, deallocator: @escaping () -> Void) {
      self.pointer = pointer
      self.count = count
      self.deallocator = deallocator
    }

    func deepCopy() -> Buffer {
      let new = Buffer(count: count)
      if count > 0 {
        new.pointer.update(from: pointer, count: count)
      }
      return new
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
  }

  // MARK: - Storage

  private var _buffer: Buffer

  /// Raw pointer to the contiguous storage.
  /// For read access this returns the current buffer's pointer directly.
  /// Write access through subscript or `withUnsafeMutableBufferPointer` triggers COW automatically.
  public var pointer: Pointer {
    _buffer.pointer
  }

  /// Number of Tensor.Scalar elements in this storage.
  public var count: Int { _buffer.count }

  // MARK: - COW

  /// Ensures this instance has exclusive ownership of its buffer.
  /// If the buffer is shared with another TensorStorage, a deep copy is made.
  @inline(__always)
  private func copyBufferIfShared() {
    if !isKnownUniquelyReferenced(&_buffer) {
      _buffer = _buffer.deepCopy()
    }
  }

  // MARK: - Initializers

  /// Allocates zeroed storage for `count` elements.
  public init(count: Int) {
    self._buffer = Buffer(count: count)
  }

  /// Allocates storage filled with `repeating` value.
  public init(repeating value: Tensor.Scalar, count: Int) {
    self._buffer = Buffer(repeating: value, count: count)
  }

  /// Allocates storage and copies data from an `Array`.
  public init(_ array: [Tensor.Scalar]) {
    let n = array.count
    if n == 0 {
      self._buffer = Buffer(count: 0)
    } else {
      let buf = Buffer(count: n)
      array.withUnsafeBufferPointer { src in
        buf.pointer.update(from: src.baseAddress!, count: n)
      }
      self._buffer = buf
    }
  }

  /// Allocates storage and copies data from a `ContiguousArray`.
  public init(_ contiguous: ContiguousArray<Tensor.Scalar>) {
    let n = contiguous.count
    if n == 0 {
      self._buffer = Buffer(count: 0)
    } else {
      let buf = Buffer(count: n)
      contiguous.withUnsafeBufferPointer { src in
        buf.pointer.update(from: src.baseAddress!, count: n)
      }
      self._buffer = buf
    }
  }

  /// External-memory initializer for subclasses (e.g. MTLBuffer-backed storage).
  public init(pointer: UnsafeMutablePointer<Tensor.Scalar>, count: Int, deallocator: @escaping () -> Void) {
    self._buffer = Buffer(pointer: pointer, count: count, deallocator: deallocator)
  }

  /// Private initializer that shares an existing buffer (for COW copy).
  private init(buffer: Buffer) {
    self._buffer = buffer
  }
  
  // MARK: - Factory (Metal-backed when NEURON_USE_METAL_STORAGE is set)

  /// Creates storage for `count` elements.
  /// Uses MetalTensorStorage when Metal is available and `NEURON_USE_METAL_STORAGE` is defined.
  /// Default: CPU-backed storage for best performance until GPU compute is implemented.
  public static func create(count: Int) -> TensorStorage {
  
    if DeviceManager.shared.type == .gpu,
        let device = MetalContext.shared.device {
      let pool = MetalContext.shared.bufferPool
      let storage = MetalTensorStorage(device: device, count: count, pool: pool)
      return storage
    }
    
    return TensorStorage(count: count)
  }

  /// Creates storage by copying `array`.
  public static func create(from array: [Tensor.Scalar]) -> TensorStorage {
    
    if DeviceManager.shared.type == .gpu,
       let device = MetalContext.shared.device {
      let pool = MetalContext.shared.bufferPool
      return MetalTensorStorage(device: device, data: array, pool: pool)
    }
    
    return TensorStorage(array)
  }

  /// Creates storage by copying `array`.
  public static func create(from contiguous: ContiguousArray<Tensor.Scalar>) -> TensorStorage {
    
    if DeviceManager.shared.type == .gpu,
       let device = MetalContext.shared.device {
      let pool = MetalContext.shared.bufferPool
      return MetalTensorStorage(device: device, data: contiguous, pool: pool)
    }
    
    return TensorStorage(contiguous)
  }

  /// Creates storage filled with `repeating` value.
  public static func create(repeating value: Tensor.Scalar, count: Int) -> TensorStorage {
    
    if DeviceManager.shared.type == .gpu,
       let device = MetalContext.shared.device {
      let pool = MetalContext.shared.bufferPool
      let storage = MetalTensorStorage(device: device, count: count, pool: pool)
      if count > 0 {
        for i in 0..<count { storage[i] = value }
      }
      return storage
    }
    
    return TensorStorage(repeating: value, count: count)
  }

  // MARK: - Codable

  /// Decodes tensor storage from a single-value JSON array.
  ///
  /// - Parameter decoder: The decoder to read scalar data from.
  /// - Throws: An error if the array cannot be decoded.
  public required convenience init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    let array = try container.decode([Tensor.Scalar].self)
    self.init(array)
  }

  // MARK: - Subscript

  /// Accesses the scalar element at the given linear index.
  /// Performs a copy-on-write if the buffer is shared before a write.
  /// - Parameter index: The zero-based position of the element to access.
  public subscript(index: Int) -> Tensor.Scalar {
    get {
      assert(index >= 0 && index < count, "TensorStorage index \(index) out of range [0..<\(count)]")
      return _buffer.pointer[index]
    }
    set {
      assert(index >= 0 && index < count, "TensorStorage index \(index) out of range [0..<\(count)]")
      copyBufferIfShared()
      _buffer.pointer[index] = newValue
    }
  }

  /// Safe range subscript returning an array, clamping to valid bounds.
  public subscript(safe range: Range<Int>, defaultValue: Tensor.Scalar) -> [Tensor.Scalar] {
    let clampedLower = Swift.max(range.lowerBound, 0)
    let clampedUpper = Swift.min(range.upperBound, count)
    guard clampedLower < clampedUpper else {
      return [Tensor.Scalar](repeating: defaultValue, count: range.count)
    }
    var result = [Tensor.Scalar](repeating: defaultValue, count: range.count)
    let validCount = clampedUpper - clampedLower
    let offset = clampedLower - range.lowerBound
    for i in 0..<validCount {
      result[offset + i] = _buffer.pointer[clampedLower + i]
    }
    return result
  }

  /// Safe single-element subscript with default.
  public subscript(safe index: Int, defaultValue: Tensor.Scalar) -> Tensor.Scalar {
    guard index >= 0 && index < count else { return defaultValue }
    return _buffer.pointer[index]
  }

  // MARK: - Properties

  /// A Boolean value indicating whether the storage contains no elements.
  public var isEmpty: Bool { count == 0 }

  /// Returns the index and value of the maximum element. Returns `(0, 0)` when empty.
  public var indexOfMax: (UInt, Tensor.Scalar) {
    guard count > 0 else { return (0, 0) }
    var maxIdx = 0
    var maxVal = _buffer.pointer[0]
    for i in 1..<count {
      let v = _buffer.pointer[i]
      if v > maxVal {
        maxVal = v
        maxIdx = i
      }
    }
    return (UInt(maxIdx), maxVal)
  }

  // MARK: - Bridging

  /// Copies data out to a `ContiguousArray<Tensor.Scalar>`.
  public func toContiguousArray() -> ContiguousArray<Tensor.Scalar> {
    guard count > 0 else { return ContiguousArray<Tensor.Scalar>() }
    return ContiguousArray<Tensor.Scalar>(UnsafeBufferPointer(start: _buffer.pointer, count: count))
  }

  /// Copies data out to an `[Tensor.Scalar]`.
  public func toArray() -> [Tensor.Scalar] {
    guard count > 0 else { return [] }
    return Array(UnsafeBufferPointer(start: _buffer.pointer, count: count))
  }

  // MARK: - Unsafe Access

  /// Calls `body` with an `UnsafeBufferPointer` over the stored scalars.
  /// - Parameter body: A closure that receives the read-only buffer pointer and returns a value.
  /// - Returns: The value returned by `body`.
  /// - Throws: Rethrows any error thrown by `body`.
  @discardableResult
  public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Tensor.Scalar>) throws -> R) rethrows -> R {
    try body(UnsafeBufferPointer(start: _buffer.pointer, count: count))
  }

  /// Calls `body` with a mutable `UnsafeMutableBufferPointer` over the stored scalars.
  ///
  /// Triggers a copy-on-write if the buffer is currently shared.
  /// - Parameter body: A closure that receives the mutable buffer pointer and returns a value.
  /// - Returns: The value returned by `body`.
  /// - Throws: Rethrows any error thrown by `body`.
  @discardableResult
  public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Tensor.Scalar>) throws -> R) rethrows -> R {
    copyBufferIfShared()
    return try body(UnsafeMutableBufferPointer(start: _buffer.pointer, count: count))
  }

  // MARK: - Copy

  /// Returns a new `TensorStorage` sharing the same underlying memory (O(1)).
  /// The actual deep copy is deferred until the first mutation on either instance.
  public func copy() -> TensorStorage {
    TensorStorage(buffer: _buffer)
  }

  /// Always allocates new memory and copies data, bypassing COW.
  public func forceCopy() -> TensorStorage {
    TensorStorage(buffer: _buffer.deepCopy())
  }
}

// MARK: - Equatable

extension TensorStorage: Equatable {
  /// Returns `true` if both storages contain the same number of elements with identical values.
  /// Performs a fast pointer-equality check before falling back to `memcmp`.
  /// - Parameter lhs: The left-hand side `TensorStorage` to compare.
  /// - Parameter rhs: The right-hand side `TensorStorage` to compare.
  /// - Returns: `true` if the two storages are element-wise equal, otherwise `false`.
  public static func == (lhs: TensorStorage, rhs: TensorStorage) -> Bool {
    guard lhs.count == rhs.count else { return false }
    if lhs._buffer === rhs._buffer { return true }
    guard lhs.count > 0 else { return true }
    return memcmp(lhs._buffer.pointer, rhs._buffer.pointer, lhs.count * MemoryLayout<Tensor.Scalar>.stride) == 0
  }
}

// MARK: - Sequence & Collection

extension TensorStorage: Sequence {
  /// Returns an iterator over the scalar elements of the storage.
  /// - Returns: A `TensorStorageIterator` positioned before the first element.
  public func makeIterator() -> TensorStorageIterator {
    TensorStorageIterator(storage: self)
  }

  /// An iterator that traverses the scalar elements of a `TensorStorage` sequentially.
  public struct TensorStorageIterator: IteratorProtocol {
    private let storage: TensorStorage
    private var index: Int = 0

    init(storage: TensorStorage) {
      self.storage = storage
    }

    /// Advances to the next scalar element and returns it, or returns `nil` when exhausted.
    /// - Returns: The next `Tensor.Scalar` value, or `nil` if the iterator has been exhausted.
    public mutating func next() -> Tensor.Scalar? {
      guard index < storage.count else { return nil }
      let value = storage._buffer.pointer[index]
      index += 1
      return value
    }
  }
}

extension TensorStorage: RandomAccessCollection, MutableCollection {
  /// The index of the first element in the collection.
  public var startIndex: Int { 0 }
  /// The index one past the last element in the collection.
  public var endIndex: Int { count }

  /// Returns the index immediately after the given index.
  /// - Parameter i: A valid index of the collection.
  /// - Returns: The index immediately after `i`.
  public func index(after i: Int) -> Int { i + 1 }
  /// Returns the index immediately before the given index.
  /// - Parameter i: A valid index of the collection.
  /// - Returns: The index immediately before `i`.
  public func index(before i: Int) -> Int { i - 1 }
}

// MARK: - Codable

extension TensorStorage: Codable {
  /// Encodes the storage's scalar elements as a single-value array into the given encoder.
  /// - Parameter encoder: The encoder to write data to.
  /// - Throws: An error if encoding fails.
  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(toArray())
  }
}

// MARK: - CustomDebugStringConvertible

extension TensorStorage: CustomDebugStringConvertible {
  /// A textual representation of the storage showing up to the first eight elements
  /// and the total count when the storage contains more than eight elements.
  public var debugDescription: String {
    let preview = Swift.min(count, 8)
    let elements = (0..<preview).map { String(describing: _buffer.pointer[$0]) }
    let suffix = count > preview ? ", ... (\(count) total)" : ""
    return "TensorStorage[\(count)](\(elements.joined(separator: ", "))\(suffix))"
  }
}

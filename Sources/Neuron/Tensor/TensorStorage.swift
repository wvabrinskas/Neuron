//
//  TensorStorage.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/4/26.
//

import Foundation

/// A pointer-backed contiguous memory container for tensor data with copy-on-write semantics.
///
/// `TensorStorage` manages a block of `Scalar` values via raw pointers,
/// providing subscript access, Collection conformance, and Codable support.
/// Multiple `TensorStorage` instances can share the same underlying memory;
/// the actual copy is deferred until the first mutation (copy-on-write).
///
/// Memory lifecycle: the base class allocates via `UnsafeMutablePointer.allocate(capacity:)`
/// and deallocates when the last reference to the inner buffer is released.
/// Subclasses that provide externally-managed memory should use the
/// `init(pointer:count:deallocator:)` initializer with a custom deallocator.
public class TensorStorage {
  #if QUANTIZED_F16
  public typealias Scalar = Float16
  #else
  public typealias Scalar = Float
  #endif

  // MARK: - Inner Buffer (reference-counted memory owner)

  final class Buffer {
    var pointer: UnsafeMutablePointer<Scalar>
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

    init(repeating value: Scalar, count: Int) {
      self.count = count
      self.deallocator = nil
      if count > 0 {
        self.pointer = .allocate(capacity: count)
        self.pointer.initialize(repeating: value, count: count)
      } else {
        self.pointer = .allocate(capacity: 1)
      }
    }

    init(pointer: UnsafeMutablePointer<Scalar>, count: Int, deallocator: @escaping () -> Void) {
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
  public var pointer: UnsafeMutablePointer<Scalar> {
    _buffer.pointer
  }

  /// Number of scalar elements in this storage.
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
  public init(repeating value: Scalar, count: Int) {
    self._buffer = Buffer(repeating: value, count: count)
  }

  /// Allocates storage and copies data from an `Array`.
  public init(_ array: [Scalar]) {
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
  public init(_ contiguous: ContiguousArray<Scalar>) {
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
  public init(pointer: UnsafeMutablePointer<Scalar>, count: Int, deallocator: @escaping () -> Void) {
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
    #if NEURON_USE_METAL_STORAGE
    if let device = MetalContext.shared.device {
      let pool = MetalContext.shared.bufferPool
      let storage = MetalTensorStorage(device: device, count: count, pool: pool)
      return storage
    }
    #endif
    return TensorStorage(count: count)
  }

  /// Creates storage by copying `array`.
  public static func create(from array: [Scalar]) -> TensorStorage {
    #if NEURON_USE_METAL_STORAGE
    if let device = MetalContext.shared.device {
      let pool = MetalContext.shared.bufferPool
      return MetalTensorStorage(device: device, data: array, pool: pool)
    }
    #endif
    return TensorStorage(array)
  }

  /// Creates storage by copying `contiguous`.
  public static func create(from contiguous: ContiguousArray<Scalar>) -> TensorStorage {
    create(from: Array(contiguous))
  }

  /// Creates storage filled with `repeating` value.
  public static func create(repeating value: Scalar, count: Int) -> TensorStorage {
    #if NEURON_USE_METAL_STORAGE
    if let device = MetalContext.shared.device {
      let pool = MetalContext.shared.bufferPool
      let storage = MetalTensorStorage(device: device, count: count, pool: pool)
      if count > 0 {
        for i in 0..<count { storage[i] = value }
      }
      return storage
    }
    #endif
    return TensorStorage(repeating: value, count: count)
  }

  // MARK: - Codable

  public required convenience init(from decoder: any Decoder) throws {
    let container = try decoder.singleValueContainer()
    let array = try container.decode([Scalar].self)
    self.init(array)
  }

  // MARK: - Subscript

  public subscript(index: Int) -> Scalar {
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
  public subscript(safe range: Range<Int>, defaultValue: Scalar) -> [Scalar] {
    let clampedLower = Swift.max(range.lowerBound, 0)
    let clampedUpper = Swift.min(range.upperBound, count)
    guard clampedLower < clampedUpper else {
      return [Scalar](repeating: defaultValue, count: range.count)
    }
    var result = [Scalar](repeating: defaultValue, count: range.count)
    let validCount = clampedUpper - clampedLower
    let offset = clampedLower - range.lowerBound
    for i in 0..<validCount {
      result[offset + i] = _buffer.pointer[clampedLower + i]
    }
    return result
  }

  /// Safe single-element subscript with default.
  public subscript(safe index: Int, defaultValue: Scalar) -> Scalar {
    guard index >= 0 && index < count else { return defaultValue }
    return _buffer.pointer[index]
  }

  // MARK: - Properties

  public var isEmpty: Bool { count == 0 }

  // MARK: - Bridging

  /// Copies data out to a `ContiguousArray<Scalar>`.
  public func toContiguousArray() -> ContiguousArray<Scalar> {
    guard count > 0 else { return ContiguousArray<Scalar>() }
    return ContiguousArray<Scalar>(UnsafeBufferPointer(start: _buffer.pointer, count: count))
  }

  /// Copies data out to an `[Scalar]`.
  public func toArray() -> [Scalar] {
    guard count > 0 else { return [] }
    return Array(UnsafeBufferPointer(start: _buffer.pointer, count: count))
  }

  // MARK: - Unsafe Access

  /// Calls `body` with a read-only `UnsafeBufferPointer` covering the entire storage.
  @discardableResult
  public func withUnsafeBufferPointer<R>(_ body: (UnsafeBufferPointer<Scalar>) throws -> R) rethrows -> R {
    try body(UnsafeBufferPointer(start: _buffer.pointer, count: count))
  }

  /// Calls `body` with a mutable `UnsafeMutableBufferPointer`.
  /// Triggers COW if the buffer is shared.
  @discardableResult
  public func withUnsafeMutableBufferPointer<R>(_ body: (UnsafeMutableBufferPointer<Scalar>) throws -> R) rethrows -> R {
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
  public static func == (lhs: TensorStorage, rhs: TensorStorage) -> Bool {
    guard lhs.count == rhs.count else { return false }
    if lhs._buffer === rhs._buffer { return true }
    guard lhs.count > 0 else { return true }
    return memcmp(lhs._buffer.pointer, rhs._buffer.pointer, lhs.count * MemoryLayout<Scalar>.stride) == 0
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
      let value = storage._buffer.pointer[index]
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
  public func encode(to encoder: any Encoder) throws {
    var container = encoder.singleValueContainer()
    try container.encode(toArray())
  }
}

// MARK: - CustomDebugStringConvertible

extension TensorStorage: CustomDebugStringConvertible {
  public var debugDescription: String {
    let preview = Swift.min(count, 8)
    let elements = (0..<preview).map { String(describing: _buffer.pointer[$0]) }
    let suffix = count > preview ? ", ... (\(count) total)" : ""
    return "TensorStorage[\(count)](\(elements.joined(separator: ", "))\(suffix))"
  }
}

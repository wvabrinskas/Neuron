//
//  BufferPool.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/5/26.
//

import Foundation
import Metal

/// Thread-safe pool of MTLBuffers, bucketed by size (next power of 2).
///
/// Reduces allocation overhead by reusing buffers of similar size.
/// Buffers are acquired with `acquire(byteCount:)` and returned with `release(_:)`.
public final class BufferPool {

  private var buckets: [Int: [MTLBuffer]] = [:]
  private let lock = NSLock()
  private weak var device: MTLDevice?

  /// Creates a buffer pool for the given Metal device.
  public init(device: MTLDevice) {
    self.device = device
  }

  /// Returns the smallest power of 2 >= `byteCount`, with a minimum of 1.
  private func bucketSize(for byteCount: Int) -> Int {
    guard byteCount > 0 else { return 1 }
    var n = 1
    while n < byteCount { n *= 2 }
    return n
  }

  /// Acquires a buffer of at least `byteCount` bytes.
  /// Reuses a pooled buffer if available; otherwise allocates a new one.
  /// - Parameter byteCount: Minimum byte length required.
  /// - Returns: An MTLBuffer with `.storageModeShared`, or `nil` if device is unavailable.
  public func acquire(byteCount: Int) -> MTLBuffer? {
    guard let device else { return nil }
    let size = bucketSize(for: byteCount)

    lock.lock()
    defer { lock.unlock() }

    if var available = buckets[size], !available.isEmpty {
      let buffer = available.removeLast()
      buckets[size] = available.isEmpty ? nil : available
      return buffer
    }

    return device.makeBuffer(length: size, options: .storageModeShared)
  }

  /// Returns a buffer to the pool for reuse.
  /// - Parameter buffer: The buffer to release. Its length is used for bucketing.
  public func release(_ buffer: MTLBuffer) {
    lock.lock()
    defer { lock.unlock() }

    let bucket = bucketSize(for: buffer.length)
    var available = buckets[bucket] ?? []
    available.append(buffer)
    buckets[bucket] = available
  }

  /// Removes all cached buffers from the pool.
  public func drain() {
    lock.lock()
    defer { lock.unlock() }
    buckets.removeAll()
  }
}

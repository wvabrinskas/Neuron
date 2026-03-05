//
//  MetalTensorStorage.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/5/26.
//

import Foundation
import Metal

/// MTLBuffer-backed tensor storage for zero-copy GPU access on Apple Silicon.
///
/// Uses `.storageModeShared` so the same memory is accessible from both CPU and GPU.
/// CPU access via `pointer` is zero-copy; the MTLBuffer's contents are directly addressable.
/// Inherits all arithmetic and Collection behavior from `TensorStorage`.
public final class MetalTensorStorage: TensorStorage {

  /// The Metal buffer backing this storage. Retained for the lifetime of the instance.
  public let mtlBuffer: MTLBuffer

  /// Allocates zeroed MTLBuffer-backed storage for `count` elements.
  /// Uses BufferPool when available for buffer recycling.
  /// - Parameters:
  ///   - device: The Metal device to allocate from.
  ///   - count: Number of scalar elements.
  ///   - pool: Optional buffer pool for acquiring buffers. When nil, allocates directly.
  public init(device: MTLDevice, count: Int, pool: BufferPool? = nil) {
    let byteCount = Swift.max(count * MemoryLayout<Tensor.Scalar>.stride, 1)
    let buffer: MTLBuffer
    if let pool, let acquired = pool.acquire(byteCount: byteCount) {
      buffer = acquired
    } else {
      buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
    }
    self.mtlBuffer = buffer
    let ptr = buffer.contents().assumingMemoryBound(to: Tensor.Scalar.self)
    super.init(pointer: ptr, count: count) { [buffer, pool] in
      pool?.release(buffer)
      _ = buffer
    }
    if count > 0 {
      ptr.initialize(repeating: 0, count: count)
    }
  }

  /// Allocates MTLBuffer-backed storage and copies data from an array.
  public init(device: MTLDevice, data: [Tensor.Scalar], pool: BufferPool? = nil) {
    let count = data.count
    let byteCount = Swift.max(count * MemoryLayout<Tensor.Scalar>.stride, 1)
    let buffer: MTLBuffer
    if let pool, let acquired = pool.acquire(byteCount: byteCount) {
      buffer = acquired
    } else {
      buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
    }
    self.mtlBuffer = buffer
    let ptr = buffer.contents().assumingMemoryBound(to: Tensor.Scalar.self)
    super.init(pointer: ptr, count: count) { [buffer, pool] in
      pool?.release(buffer)
      _ = buffer
    }
    if count > 0 {
      data.withUnsafeBufferPointer { src in
        ptr.initialize(from: src.baseAddress!, count: count)
      }
    }
  }

  /// Allocates MTLBuffer-backed storage and copies data from existing TensorStorage.
  public init(device: MTLDevice, storage: TensorStorage, pool: BufferPool? = nil) {
    let count = storage.count
    let byteCount = Swift.max(count * MemoryLayout<Tensor.Scalar>.stride, 1)
    let buffer: MTLBuffer
    if let pool, let acquired = pool.acquire(byteCount: byteCount) {
      buffer = acquired
    } else {
      buffer = device.makeBuffer(length: byteCount, options: .storageModeShared)!
    }
    self.mtlBuffer = buffer
    let ptr = buffer.contents().assumingMemoryBound(to: Tensor.Scalar.self)
    super.init(pointer: ptr, count: count) { [buffer, pool] in
      pool?.release(buffer)
      _ = buffer
    }
    if count > 0 {
      ptr.initialize(from: storage.pointer, count: count)
    }
  }

  required convenience init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    let array = try container.decode([Tensor.Scalar].self)
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw DecodingError.dataCorrupted(
        DecodingError.Context(codingPath: decoder.codingPath,
                              debugDescription: "Metal not available for MetalTensorStorage decoding"))
    }
    self.init(device: device, data: array)
  }
}

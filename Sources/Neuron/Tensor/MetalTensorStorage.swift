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
  /// - Parameters:
  ///   - device: The Metal device to allocate from.
  ///   - count: Number of scalar elements.
  public init(device: MTLDevice, count: Int) {
    let byteCount = count * MemoryLayout<Scalar>.stride
    let buffer = device.makeBuffer(length: Swift.max(byteCount, 1), options: .storageModeShared)!
    self.mtlBuffer = buffer
    let ptr = buffer.contents().assumingMemoryBound(to: Scalar.self)
    super.init(pointer: ptr, count: count) { [buffer] in
      _ = buffer
    }
    if count > 0 {
      ptr.initialize(repeating: 0, count: count)
    }
  }

  /// Allocates MTLBuffer-backed storage and copies data from an array.
  public init(device: MTLDevice, data: [Scalar]) {
    let count = data.count
    let byteCount = count * MemoryLayout<Scalar>.stride
    let buffer = device.makeBuffer(length: Swift.max(byteCount, 1), options: .storageModeShared)!
    self.mtlBuffer = buffer
    let ptr = buffer.contents().assumingMemoryBound(to: Scalar.self)
    super.init(pointer: ptr, count: count) { [buffer] in
      _ = buffer
    }
    if count > 0 {
      data.withUnsafeBufferPointer { src in
        ptr.initialize(from: src.baseAddress!, count: count)
      }
    }
  }

  /// Allocates MTLBuffer-backed storage and copies data from existing TensorStorage.
  public init(device: MTLDevice, storage: TensorStorage) {
    let count = storage.count
    let byteCount = count * MemoryLayout<Scalar>.stride
    let buffer = device.makeBuffer(length: Swift.max(byteCount, 1), options: .storageModeShared)!
    self.mtlBuffer = buffer
    let ptr = buffer.contents().assumingMemoryBound(to: Scalar.self)
    super.init(pointer: ptr, count: count) { [buffer] in
      _ = buffer
    }
    if count > 0 {
      ptr.initialize(from: storage.pointer, count: count)
    }
  }

  required convenience init(from decoder: Decoder) throws {
    let container = try decoder.singleValueContainer()
    let array = try container.decode([Scalar].self)
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw DecodingError.dataCorrupted(
        DecodingError.Context(codingPath: decoder.codingPath,
                              debugDescription: "Metal not available for MetalTensorStorage decoding"))
    }
    self.init(device: device, data: array)
  }
}

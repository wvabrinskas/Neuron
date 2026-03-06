//
//  BatchLayout.swift
//  Neuron
//
//  NCHW packing and unpacking for batched GPU execution.
//  Neuron uses (columns, rows, depth) = (W, H, C) with CHW storage per sample.
//  Metal expects NCHW; packing = concatenate samples [s0 | s1 | ... | sN].
//

import Foundation
import Metal

/// Utilities for packing and unpacking tensor batches in NCHW layout for Metal.
public enum BatchLayout {

  /// Packs a batch of tensors into a single MetalTensorStorage in NCHW layout.
  ///
  /// Neuron layout is CHW per sample (depth, rows, columns). NCHW concatenates
  /// samples: [s0 | s1 | ... | sN]. No reorder needed.
  ///
  /// - Parameters:
  ///   - batch: Array of tensors with identical TensorSize.
  ///   - device: Metal device for allocation.
  ///   - pool: Optional buffer pool for buffer recycling.
  /// - Returns: Single MetalTensorStorage containing all samples, or nil if Metal unavailable.
  public static func packToNCHW(
    _ batch: [Tensor],
    device: MTLDevice,
    pool: BufferPool?
  ) -> MetalTensorStorage? {
    guard let first = batch.first, !batch.isEmpty else { return nil }
    let singleSize = first.size
    let elementsPerSample = singleSize.rows * singleSize.columns * singleSize.depth
    let totalCount = batch.count * elementsPerSample

    let storage = MetalTensorStorage(device: device, count: totalCount, pool: pool)
    let outPtr = storage.pointer

    for (i, tensor) in batch.enumerated() {
      precondition(
        tensor.size == singleSize,
        "BatchLayout.packToNCHW: all tensors must have identical size. Expected \(singleSize), got \(tensor.size) at index \(i)"
      )
      let offset = i * elementsPerSample
      outPtr.advanced(by: offset).update(from: tensor.storage.pointer, count: elementsPerSample)
    }

    return storage
  }

  /// Unpacks a MetalTensorStorage in NCHW layout into per-sample tensors.
  ///
  /// - Parameters:
  ///   - storage: Batched Metal storage in NCHW [N, C, H, W].
  ///   - batchCount: Number of samples (N).
  ///   - singleSize: Per-sample TensorSize (columns, rows, depth).
  ///   - device: Metal device for creating output storage.
  ///   - pool: Optional buffer pool.
  ///   - context: Optional TensorContext to attach to each output (e.g. backward closure).
  /// - Returns: Array of tensors, one per sample.
  public static func unpackFromNCHW(
    _ storage: MetalTensorStorage,
    batchCount: Int,
    singleSize: TensorSize,
    device: MTLDevice,
    pool: BufferPool?,
    context: TensorContext = TensorContext()
  ) -> [Tensor] {
    let elementsPerSample = singleSize.rows * singleSize.columns * singleSize.depth
    precondition(
      storage.count >= batchCount * elementsPerSample,
      "BatchLayout.unpackFromNCHW: storage has \(storage.count) elements, need \(batchCount * elementsPerSample)"
    )

    var result: [Tensor] = []
    result.reserveCapacity(batchCount)

    for i in 0..<batchCount {
      let offset = i * elementsPerSample
      let slice = Tensor.Value(unsafeUninitializedCapacity: elementsPerSample) { buf, count in
        buf.baseAddress!.initialize(from: storage.pointer.advanced(by: offset), count: elementsPerSample)
        count = elementsPerSample
      }
      let sliceStorage = MetalTensorStorage(device: device, data: slice, pool: pool)
      let tensor = Tensor(storage: sliceStorage, size: singleSize, context: context)
      result.append(tensor)
    }

    return result
  }
}

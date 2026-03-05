//
//  MetalEngine.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/5/26.
//

import Foundation
import Metal

/// Metal compute dispatch layer for neural network operations.
///
/// Loads and caches compute pipeline states from the default Metal library.
/// Uses MetalContext for device, command queue, and buffer pool.
/// Designed for single-command-buffer encoding (encoder passthrough in later commits).
public final class MetalEngine {

  private var pipelineCache: [String: MTLComputePipelineState] = [:]
  private let cacheLock = NSLock()

  /// Creates an engine using MetalContext.shared.
  public init() {}

  /// Whether Metal is available and the engine can dispatch.
  public var isAvailable: Bool { MetalContext.shared.isAvailable }

  /// The Metal device, or `nil` if unavailable.
  public var device: MTLDevice? { MetalContext.shared.device }

  /// The command queue for creating command buffers.
  public var commandQueue: MTLCommandQueue? { MetalContext.shared.commandQueue }

  /// The buffer pool for acquiring and releasing MTLBuffers.
  public var bufferPool: BufferPool? { MetalContext.shared.bufferPool }

  /// Loads and caches a compute pipeline by kernel name.
  ///
  /// - Parameter name: The kernel function name (e.g. `"neuron_activation"`).
  /// - Returns: The cached or newly created pipeline state, or `nil` if Metal is unavailable or the kernel is not found.
  public func pipeline(named name: String) -> MTLComputePipelineState? {
    guard let device else { return nil }

    cacheLock.lock()
    if let cached = pipelineCache[name] {
      cacheLock.unlock()
      return cached
    }
    cacheLock.unlock()

    guard let library = device.makeDefaultLibrary(),
          let function = library.makeFunction(name: name) else {
      return nil
    }

    do {
      let pipeline = try device.makeComputePipelineState(function: function)
      cacheLock.lock()
      pipelineCache[name] = pipeline
      cacheLock.unlock()
      return pipeline
    } catch {
      return nil
    }
  }

  /// Acquires a buffer of at least `byteCount` bytes from the pool.
  ///
  /// - Parameter byteCount: Minimum byte length required.
  /// - Returns: An MTLBuffer with `.storageModeShared`, or `nil` if the pool or device is unavailable.
  public func acquireBuffer(byteCount: Int) -> MTLBuffer? {
    bufferPool?.acquire(byteCount: byteCount)
  }

  /// Returns a buffer to the pool for reuse.
  ///
  /// - Parameter buffer: The buffer to release.
  public func releaseBuffer(_ buffer: MTLBuffer) {
    bufferPool?.release(buffer)
  }

  /// Creates a new command buffer from the shared command queue.
  ///
  /// - Returns: A new MTLCommandBuffer, or `nil` if the queue is unavailable.
  public func makeCommandBuffer() -> MTLCommandBuffer? {
    commandQueue?.makeCommandBuffer()
  }

  // MARK: - Activation dispatch

  /// Dispatches the neuron_activation kernel.
  ///
  /// - Parameters:
  ///   - input: Input MetalTensorStorage (read-only).
  ///   - output: Output MetalTensorStorage (written by kernel).
  ///   - activationType: Activation index (0–8, see Activation.index()).
  ///   - leakyAlpha: Alpha for leakyRelu; ignored for other types.
  /// - Returns: `true` if dispatch succeeded, `false` otherwise.
  public func dispatchActivation(
    input: MetalTensorStorage,
    output: MetalTensorStorage,
    activationType: UInt32,
    leakyAlpha: Float
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_activation"),
          let cmdBuffer = makeCommandBuffer(),
          let encoder = cmdBuffer.makeComputeCommandEncoder() else {
      return false
    }

    let count = input.count
    guard count > 0, count == output.count else { return false }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input.mtlBuffer, offset: 0, index: 0)
    encoder.setBuffer(output.mtlBuffer, offset: 0, index: 1)

    var actType = activationType
    var alpha = leakyAlpha
    encoder.setBytes(&actType, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 3)

    let tgSize = MTLSize(width: min(256, max(1, count)), height: 1, depth: 1)
    let gridSize = MTLSize(width: count, height: 1, depth: 1)
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    encoder.endEncoding()
    cmdBuffer.commit()
    cmdBuffer.waitUntilCompleted()

    return cmdBuffer.status == .completed
  }

  /// Dispatches the neuron_derivate kernel.
  ///
  /// - Parameters:
  ///   - input: Input MetalTensorStorage (read-only).
  ///   - output: Output MetalTensorStorage (written by kernel).
  ///   - activationType: Activation index (0–8).
  ///   - leakyAlpha: Alpha for leakyRelu; ignored for other types.
  /// - Returns: `true` if dispatch succeeded, `false` otherwise.
  public func dispatchDerivate(
    input: MetalTensorStorage,
    output: MetalTensorStorage,
    activationType: UInt32,
    leakyAlpha: Float
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_derivate"),
          let cmdBuffer = makeCommandBuffer(),
          let encoder = cmdBuffer.makeComputeCommandEncoder() else {
      return false
    }

    let count = input.count
    guard count > 0, count == output.count else { return false }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input.mtlBuffer, offset: 0, index: 0)
    encoder.setBuffer(output.mtlBuffer, offset: 0, index: 1)

    var actType = activationType
    var alpha = leakyAlpha
    encoder.setBytes(&actType, length: MemoryLayout<UInt32>.size, index: 2)
    encoder.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 3)

    let tgSize = MTLSize(width: min(256, max(1, count)), height: 1, depth: 1)
    let gridSize = MTLSize(width: count, height: 1, depth: 1)
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    encoder.endEncoding()
    cmdBuffer.commit()
    cmdBuffer.waitUntilCompleted()

    return cmdBuffer.status == .completed
  }

  // MARK: - Matrix multiplication dispatch

  /// Dispatches the neuron_matmul_tiled kernel.
  /// C[M×N] = A[M×K] × B[K×N], row-major layout.
  /// For depth > 1, dispatches once per depth slice with buffer offsets.
  ///
  /// - Parameters:
  ///   - a: Left matrix storage (M×K per slice).
  ///   - b: Right matrix storage (K×N per slice).
  ///   - output: Output storage (M×N per slice).
  ///   - M: Rows of A / rows of C.
  ///   - N: Columns of B / columns of C.
  ///   - K: Columns of A / rows of B.
  ///   - depth: Number of depth slices; each slice uses contiguous buffer regions.
  /// - Returns: `true` if all dispatches succeeded, `false` otherwise.
  public func dispatchMatmul(
    a: MetalTensorStorage,
    b: MetalTensorStorage,
    output: MetalTensorStorage,
    M: Int, N: Int, K: Int,
    depth: Int = 1
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_matmul_tiled"),
          let cmdBuffer = makeCommandBuffer(),
          let encoder = cmdBuffer.makeComputeCommandEncoder() else {
      return false
    }

    let stride = MemoryLayout<Tensor.Scalar>.stride
    let aSliceSize = M * K
    let bSliceSize = K * N
    let cSliceSize = M * N

    guard a.count >= aSliceSize * depth,
          b.count >= bSliceSize * depth,
          output.count >= cSliceSize * depth else {
      return false
    }

    encoder.setComputePipelineState(pipeline)

    var mVal = UInt32(M)
    var nVal = UInt32(N)
    var kVal = UInt32(K)
    encoder.setBytes(&mVal, length: MemoryLayout<UInt32>.size, index: 3)
    encoder.setBytes(&nVal, length: MemoryLayout<UInt32>.size, index: 4)
    encoder.setBytes(&kVal, length: MemoryLayout<UInt32>.size, index: 5)

    let gridWidth = (N + 15) / 16
    let gridHeight = (M + 15) / 16
    let gridSize = MTLSize(width: max(1, gridWidth), height: max(1, gridHeight), depth: 1)
    let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

    for d in 0..<depth {
      let aOffset = d * aSliceSize * stride
      let bOffset = d * bSliceSize * stride
      let cOffset = d * cSliceSize * stride

      encoder.setBuffer(a.mtlBuffer, offset: aOffset, index: 0)
      encoder.setBuffer(b.mtlBuffer, offset: bOffset, index: 1)
      encoder.setBuffer(output.mtlBuffer, offset: cOffset, index: 2)

      encoder.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }

    encoder.endEncoding()
    cmdBuffer.commit()
    cmdBuffer.waitUntilCompleted()

    return cmdBuffer.status == .completed
  }
}

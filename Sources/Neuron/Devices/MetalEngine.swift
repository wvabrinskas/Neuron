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

  // MARK: - Encoder passthrough (encode into provided encoder; no commit/wait)

  /// Encodes neuron_activation into the given encoder. Does not end encoding or commit.
  public func encodeActivation(
    encoder: MetalCommandEncoder,
    input: MetalTensorStorage,
    output: MetalTensorStorage,
    activationType: UInt32,
    leakyAlpha: Float
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_activation") else { return false }
    let count = input.count
    guard count > 0, count == output.count else { return false }

    let enc = encoder.encoder
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(input.mtlBuffer, offset: 0, index: 0)
    enc.setBuffer(output.mtlBuffer, offset: 0, index: 1)
    var actType = activationType
    var alpha = leakyAlpha
    enc.setBytes(&actType, length: MemoryLayout<UInt32>.size, index: 2)
    enc.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 3)
    let tgSize = MTLSize(width: min(256, max(1, count)), height: 1, depth: 1)
    let gridSize = MTLSize(width: count, height: 1, depth: 1)
    enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    return true
  }

  /// Encodes neuron_derivate into the given encoder. Does not end encoding or commit.
  public func encodeDerivate(
    encoder: MetalCommandEncoder,
    input: MetalTensorStorage,
    output: MetalTensorStorage,
    activationType: UInt32,
    leakyAlpha: Float
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_derivate") else { return false }
    let count = input.count
    guard count > 0, count == output.count else { return false }

    let enc = encoder.encoder
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(input.mtlBuffer, offset: 0, index: 0)
    enc.setBuffer(output.mtlBuffer, offset: 0, index: 1)
    var actType = activationType
    var alpha = leakyAlpha
    enc.setBytes(&actType, length: MemoryLayout<UInt32>.size, index: 2)
    enc.setBytes(&alpha, length: MemoryLayout<Float>.size, index: 3)
    let tgSize = MTLSize(width: min(256, max(1, count)), height: 1, depth: 1)
    let gridSize = MTLSize(width: count, height: 1, depth: 1)
    enc.dispatchThreads(gridSize, threadsPerThreadgroup: tgSize)
    return true
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

  /// Encodes neuron_matmul_tiled into the given encoder. Does not end encoding or commit.
  public func encodeMatmul(
    encoder: MetalCommandEncoder,
    a: MetalTensorStorage,
    b: MetalTensorStorage,
    output: MetalTensorStorage,
    M: Int, N: Int, K: Int,
    depth: Int = 1
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_matmul_tiled") else { return false }
    let stride = MemoryLayout<Tensor.Scalar>.stride
    let aSliceSize = M * K
    let bSliceSize = K * N
    let cSliceSize = M * N
    guard a.count >= aSliceSize * depth,
          b.count >= bSliceSize * depth,
          output.count >= cSliceSize * depth else {
      return false
    }

    let enc = encoder.encoder
    enc.setComputePipelineState(pipeline)
    var mVal = UInt32(M)
    var nVal = UInt32(N)
    var kVal = UInt32(K)
    enc.setBytes(&mVal, length: MemoryLayout<UInt32>.size, index: 3)
    enc.setBytes(&nVal, length: MemoryLayout<UInt32>.size, index: 4)
    enc.setBytes(&kVal, length: MemoryLayout<UInt32>.size, index: 5)
    let gridWidth = (N + 15) / 16
    let gridHeight = (M + 15) / 16
    let gridSize = MTLSize(width: max(1, gridWidth), height: max(1, gridHeight), depth: 1)
    let threadgroupSize = MTLSize(width: 16, height: 16, depth: 1)

    for d in 0..<depth {
      let aOffset = d * aSliceSize * stride
      let bOffset = d * bSliceSize * stride
      let cOffset = d * cSliceSize * stride
      enc.setBuffer(a.mtlBuffer, offset: aOffset, index: 0)
      enc.setBuffer(b.mtlBuffer, offset: bOffset, index: 1)
      enc.setBuffer(output.mtlBuffer, offset: cOffset, index: 2)
      enc.dispatchThreadgroups(gridSize, threadsPerThreadgroup: threadgroupSize)
    }
    return true
  }

  // MARK: - Convolution dispatch

  /// Parameters for neuron_conv2d_implicit_gemm (must match Metal Conv2DParams layout).
  public struct Conv2DParams {
    public var N: UInt32
    public var C: UInt32
    public var H: UInt32
    public var W: UInt32
    public var K: UInt32
    public var kH: UInt32
    public var kW: UInt32
    public var oH: UInt32
    public var oW: UInt32
    public var strideH: UInt32
    public var strideW: UInt32
    public var padH: UInt32
    public var padW: UInt32
    public var hasBias: UInt32

    public init(N: UInt32, C: UInt32, H: UInt32, W: UInt32, K: UInt32,
                kH: UInt32, kW: UInt32, oH: UInt32, oW: UInt32,
                strideH: UInt32, strideW: UInt32, padH: UInt32, padW: UInt32,
                hasBias: UInt32) {
      self.N = N
      self.C = C
      self.H = H
      self.W = W
      self.K = K
      self.kH = kH
      self.kW = kW
      self.oH = oH
      self.oW = oW
      self.strideH = strideH
      self.strideW = strideW
      self.padH = padH
      self.padW = padW
      self.hasBias = hasBias
    }
  }

  /// Dispatches the neuron_conv2d_implicit_gemm kernel.
  /// Input [N,C,H,W], weights [K,C,kH,kW], output [N,K,oH,oW] in NCHW layout.
  ///
  /// - Parameters:
  ///   - input: Input MetalTensorStorage [N,C,H,W].
  ///   - weights: Weights MetalTensorStorage [K,C,kH,kW] concatenated.
  ///   - output: Output MetalTensorStorage [N,K,oH,oW].
  ///   - bias: Optional bias [K]; pass nil if no bias.
  ///   - params: Conv2D parameters.
  /// - Returns: `true` if dispatch succeeded, `false` otherwise.
  public func dispatchConv2d(
    input: MetalTensorStorage,
    weights: MetalTensorStorage,
    output: MetalTensorStorage,
    bias: MetalTensorStorage?,
    params: Conv2DParams
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_conv2d_implicit_gemm"),
          let cmdBuffer = makeCommandBuffer(),
          let encoder = cmdBuffer.makeComputeCommandEncoder() else {
      return false
    }

    let totalSpatial = Int(params.N) * Int(params.oH) * Int(params.oW)
    let gridWidth = Int(params.K)
    let gridHeight = totalSpatial

    guard gridWidth > 0, gridHeight > 0 else { return false }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input.mtlBuffer, offset: 0, index: 0)
    encoder.setBuffer(weights.mtlBuffer, offset: 0, index: 1)
    encoder.setBuffer(output.mtlBuffer, offset: 0, index: 2)

    var p = params
    encoder.setBytes(&p, length: MemoryLayout<Conv2DParams>.size, index: 3)

    if let bias = bias {
      encoder.setBuffer(bias.mtlBuffer, offset: 0, index: 4)
    }

    let tgWidth = min(16, gridWidth)
    let tgHeight = min(16, gridHeight)
    let threadgroupSize = MTLSize(width: tgWidth, height: tgHeight, depth: 1)
    let gridSize = MTLSize(width: gridWidth, height: gridHeight, depth: 1)
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()
    cmdBuffer.commit()
    cmdBuffer.waitUntilCompleted()

    return cmdBuffer.status == .completed
  }

  /// Encodes neuron_conv2d_implicit_gemm into the given encoder. Does not end encoding or commit.
  public func encodeConv2d(
    encoder: MetalCommandEncoder,
    input: MetalTensorStorage,
    weights: MetalTensorStorage,
    output: MetalTensorStorage,
    bias: MetalTensorStorage?,
    params: Conv2DParams
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_conv2d_implicit_gemm") else { return false }
    let totalSpatial = Int(params.N) * Int(params.oH) * Int(params.oW)
    let gridWidth = Int(params.K)
    let gridHeight = totalSpatial
    guard gridWidth > 0, gridHeight > 0 else { return false }

    let enc = encoder.encoder
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(input.mtlBuffer, offset: 0, index: 0)
    enc.setBuffer(weights.mtlBuffer, offset: 0, index: 1)
    enc.setBuffer(output.mtlBuffer, offset: 0, index: 2)
    var p = params
    enc.setBytes(&p, length: MemoryLayout<Conv2DParams>.size, index: 3)
    if let bias = bias {
      enc.setBuffer(bias.mtlBuffer, offset: 0, index: 4)
    }
    let tgWidth = min(16, gridWidth)
    let tgHeight = min(16, gridHeight)
    let threadgroupSize = MTLSize(width: tgWidth, height: tgHeight, depth: 1)
    let gridSize = MTLSize(width: gridWidth, height: gridHeight, depth: 1)
    enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    return true
  }

  /// Dispatches the neuron_conv_transpose2d kernel.
  /// Input [N,C,H,W], weights [C,K,kH,kW], output [N,K,oH,oW] in NCHW layout.
  ///
  /// - Parameters:
  ///   - input: Input MetalTensorStorage [N,C,H,W].
  ///   - weights: Weights MetalTensorStorage [C,K,kH,kW] (c-major then k).
  ///   - output: Output MetalTensorStorage [N,K,oH,oW].
  ///   - params: Conv2D parameters (hasBias ignored; no bias in transposed conv kernel).
  /// - Returns: `true` if dispatch succeeded, `false` otherwise.
  public func dispatchTransConv2d(
    input: MetalTensorStorage,
    weights: MetalTensorStorage,
    output: MetalTensorStorage,
    params: Conv2DParams
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_conv_transpose2d"),
          let cmdBuffer = makeCommandBuffer(),
          let encoder = cmdBuffer.makeComputeCommandEncoder() else {
      return false
    }

    let totalOutput = Int(params.N) * Int(params.K) * Int(params.oH) * Int(params.oW)
    guard totalOutput > 0 else { return false }

    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(input.mtlBuffer, offset: 0, index: 0)
    encoder.setBuffer(weights.mtlBuffer, offset: 0, index: 1)
    encoder.setBuffer(output.mtlBuffer, offset: 0, index: 2)

    var p = params
    encoder.setBytes(&p, length: MemoryLayout<Conv2DParams>.size, index: 3)

    let tgSize = min(256, max(1, totalOutput))
    let gridSize = MTLSize(width: totalOutput, height: 1, depth: 1)
    let threadgroupSize = MTLSize(width: tgSize, height: 1, depth: 1)
    encoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    encoder.endEncoding()
    cmdBuffer.commit()
    cmdBuffer.waitUntilCompleted()

    return cmdBuffer.status == .completed
  }

  /// Encodes neuron_conv_transpose2d into the given encoder. Does not end encoding or commit.
  public func encodeTransConv2d(
    encoder: MetalCommandEncoder,
    input: MetalTensorStorage,
    weights: MetalTensorStorage,
    output: MetalTensorStorage,
    params: Conv2DParams
  ) -> Bool {
    guard let pipeline = pipeline(named: "neuron_conv_transpose2d") else { return false }
    let totalOutput = Int(params.N) * Int(params.K) * Int(params.oH) * Int(params.oW)
    guard totalOutput > 0 else { return false }

    let enc = encoder.encoder
    enc.setComputePipelineState(pipeline)
    enc.setBuffer(input.mtlBuffer, offset: 0, index: 0)
    enc.setBuffer(weights.mtlBuffer, offset: 0, index: 1)
    enc.setBuffer(output.mtlBuffer, offset: 0, index: 2)
    var p = params
    enc.setBytes(&p, length: MemoryLayout<Conv2DParams>.size, index: 3)
    let tgSize = min(256, max(1, totalOutput))
    let gridSize = MTLSize(width: totalOutput, height: 1, depth: 1)
    let threadgroupSize = MTLSize(width: tgSize, height: 1, depth: 1)
    enc.dispatchThreads(gridSize, threadsPerThreadgroup: threadgroupSize)
    return true
  }
}

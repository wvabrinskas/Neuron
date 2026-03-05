//
//  MetalContext.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/5/26.
//

import Foundation
import Metal

/// Singleton managing Metal device and command queue for tensor operations.
///
/// Lazy initialization ensures Metal is only created when first accessed.
/// Use `isAvailable` to check before creating Metal-backed storage (e.g. in simulators).
public final class MetalContext {

  public static let shared = MetalContext()

  private let _device: MTLDevice?
  private let _commandQueue: MTLCommandQueue?
  private let _bufferPool: BufferPool?

  private init() {
    self._device = MTLCreateSystemDefaultDevice()
    self._commandQueue = _device?.makeCommandQueue()
    self._bufferPool = _device.map { BufferPool(device: $0) }
  }

  /// The system default Metal device, or `nil` if Metal is unavailable.
  public var device: MTLDevice? { _device }

  /// Shared buffer pool for MTLBuffer recycling. `nil` when Metal is unavailable.
  public var bufferPool: BufferPool? { _bufferPool }

  /// A command queue for the default device, or `nil` if Metal is unavailable.
  public var commandQueue: MTLCommandQueue? { _commandQueue }

  /// Whether Metal is usable (e.g. `false` on simulator without GPU).
  public var isAvailable: Bool { _device != nil }
}

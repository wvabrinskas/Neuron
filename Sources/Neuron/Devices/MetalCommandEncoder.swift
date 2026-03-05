//
//  MetalCommandEncoder.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/5/26.
//

import Foundation
import Metal

/// Holds a single MTLCommandBuffer and MTLComputeCommandEncoder for batched Metal encoding.
///
/// Enables encoding multiple operations (activation, matmul, conv, etc.) into one command buffer
/// before committing, reducing CPU–GPU round-trip overhead. Create at step start, pass through
/// NetworkContext, and commit once per batch.
///
/// Thread-safety: Use from a single thread; do not share across concurrent workers.
public final class MetalCommandEncoder: @unchecked Sendable {

  /// The underlying command buffer.
  public let commandBuffer: MTLCommandBuffer

  /// The compute encoder for encoding dispatches. Valid until `endEncoding()` is called.
  public let encoder: MTLComputeCommandEncoder

  /// Creates a new command buffer and compute encoder from MetalContext.
  ///
  /// - Returns: A new encoder, or `nil` if Metal is unavailable or creation fails.
  public init?() {
    guard let queue = MetalContext.shared.commandQueue,
          let cmdBuffer = queue.makeCommandBuffer(),
          let enc = cmdBuffer.makeComputeCommandEncoder() else {
      return nil
    }
    self.commandBuffer = cmdBuffer
    self.encoder = enc
  }

  /// Ends the compute encoder. Call before committing the command buffer.
  public func endEncoding() {
    encoder.endEncoding()
  }

  /// Commits the command buffer for GPU execution.
  public func commit() {
    commandBuffer.commit()
  }

  /// Blocks until the command buffer has completed execution.
  public func waitUntilCompleted() {
    commandBuffer.waitUntilCompleted()
  }

  /// Whether the command buffer finished successfully.
  public var isCompleted: Bool {
    commandBuffer.status == .completed
  }
}

//
//  MetalEncoderHolder.swift
//  Neuron
//
//  Created by William Vabrinskas on 3/5/26.
//

import Foundation
import Metal

/// Holds a Metal command encoder and supports sync-and-replace for CPU layers that must read GPU output.
///
/// When a CPU layer (e.g. InstanceNorm) needs to read from Metal-backed tensors produced by prior
/// GPU layers, the GPU must have completed. Call `syncAndReplace()` to commit the current encoder,
/// wait for completion, and create a new encoder for subsequent GPU layers.
///
/// Thread-safety: Use from a single thread; do not share across concurrent workers.
public final class MetalEncoderHolder: @unchecked Sendable {

  /// The current encoder. Replaced after `syncAndReplace()`.
  public private(set) var encoder: MetalCommandEncoder?

  /// Creates a holder with the given encoder.
  public init(encoder: MetalCommandEncoder?) {
    self.encoder = encoder
  }

  /// Commits the current encoder, waits for GPU completion, and creates a new encoder.
  /// Call before CPU layers that read from Metal-backed inputs.
  /// After this call, `encoder` is a new encoder for subsequent GPU layers.
  public func syncAndReplace() {
    guard let enc = encoder else { return }
    enc.endEncoding()
    enc.commit()
    enc.waitUntilCompleted()
    encoder = MetalCommandEncoder()
  }

  /// Commits the current encoder and waits for GPU completion. Use after forward pass
  /// before reading outputs (loss, gradients). Clears the encoder so it is not double-synced.
  public func syncAndFinish() {
    guard let enc = encoder else { return }
    enc.endEncoding()
    enc.commit()
    enc.waitUntilCompleted()
    encoder = nil
  }
}

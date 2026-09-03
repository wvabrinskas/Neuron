//
//  GradientNormInspection.swift
//  Neuron
//

import Foundation

/// L2 norm of one parameter group's gradients, captured in `Optimizer.step()` before any
/// gradient clipping or optimizer scaling.
///
/// Delivered to `BaseOptimizer.gradientNormInspector` as a diagnostic for locating which
/// parameters drive a large `Metric.globalGradientNorm`.
public struct GradientNormReport {
  /// A parameter group within a layer: its name and the L2 norms of its weight and bias gradients.
  public typealias Group = (group: String, weightNorm: Tensor.Scalar, biasNorm: Tensor.Scalar)
  
  /// Index of the layer in `Trainable.layers`.
  public let layerIndex: Int
  /// Type name of the layer, e.g. `"LSTM"` or `"Dense"`.
  public let layer: String
  /// Parameter group within the layer. Layers with a single weight tensor report `"weights"`;
  /// layers conforming to `GradientNormInspectable` report one entry per group (e.g. per LSTM gate).
  public let group: String
  /// L2 norm of the weight gradient for this group.
  public let weightNorm: Tensor.Scalar
  /// L2 norm of the bias gradient for this group.
  public let biasNorm: Tensor.Scalar
}

/// Adopted by layers that pack several parameter groups into one weight tensor so the
/// optimizer can report a per-group gradient norm breakdown instead of one number per layer.
public protocol GradientNormInspectable {
  /// Splits packed weight and bias gradients into named groups and returns each group's L2 norms.
  ///
  /// - Parameters:
  ///   - weights: The layer's packed weight gradient for this step.
  ///   - biases: The layer's packed bias gradient for this step.
  /// - Returns: One entry per parameter group, in the layer's canonical order.
  func gradientNormBreakdown(weights: Tensor, biases: Tensor) -> [GradientNormReport.Group]
}

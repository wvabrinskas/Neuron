//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/13/22.
//

import Foundation
import NumSwift

/// Represents the type of compute device available for tensor operations.
public enum DeviceType: String, Codable {
  case cpu, gpu
  
  func device() -> Device {
    switch self {
    case .cpu:
      return CPU()
    case .gpu:
      return GPU()
    }
  }
}

/// Defines the interface for a compute device capable of performing tensor operations.
///
/// Conforming types provide hardware-specific implementations of neural network
/// primitives such as convolution, activation, and pooling.
public protocol Device {
  var type: DeviceType { get }
  var qosPriority: DispatchQoS.QoSClass { get set }

  /// Computes a 2D convolution on flat tensor storage.
  ///
  /// - Parameters:
  ///   - signal: Input feature-map slice.
  ///   - filter: Convolution kernel slice.
  ///   - strides: Row/column stride.
  ///   - padding: Padding mode.
  ///   - filterSize: Kernel shape.
  ///   - inputSize: Input spatial shape.
  ///   - outputSize: Optional explicit output shape hint.
  /// - Returns: Flat convolution output.
  func conv2d(signal: Tensor.Value,
              filter: Tensor.Value,
              strides: (Int, Int),
              padding: NumSwift.ConvPadding,
              filterSize: (rows: Int, columns: Int),
              inputSize: (rows: Int, columns: Int),
              outputSize: (rows: Int, columns: Int)?) -> Tensor.Value
  
  /// Computes a transposed 2D convolution on flat tensor storage.
  ///
  /// - Parameters:
  ///   - signal: Input feature-map slice.
  ///   - filter: Transposed-convolution kernel slice.
  ///   - strides: Row/column stride.
  ///   - padding: Padding mode.
  ///   - filterSize: Kernel shape.
  ///   - inputSize: Input spatial shape.
  ///   - outputSize: Optional explicit output shape hint.
  /// - Returns: Flat transposed-convolution output.
  func transConv2d(signal: Tensor.Value,
                          filter: Tensor.Value,
                          strides: (Int, Int),
                          padding: NumSwift.ConvPadding,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)?) -> Tensor.Value
  
  /// Applies an activation function element-wise.
  ///
  /// - Parameters:
  ///   - input: Tensor to activate.
  ///   - type: Activation type to apply.
  /// - Returns: Activated tensor.
  func activate(_ input: Tensor, _ type: Activation) -> Tensor
  /// Applies an activation derivative element-wise.
  ///
  /// - Parameters:
  ///   - input: Tensor to differentiate.
  ///   - type: Activation derivative to compute.
  /// - Returns: Tensor containing derivative values.
  func derivate(_ input: Tensor, _ type: Activation) -> Tensor
  /// Computes matrix multiplication between two tensors.
  ///
  /// - Parameters:
  ///   - a: Left tensor.
  ///   - b: Right tensor.
  /// - Returns: Matrix product tensor.
  func matmul(_ a: Tensor, _ b: Tensor) -> Tensor

  /// Pointer-based 2D convolution writing into a pre-allocated result buffer.
  /// Returns `true` if the operation was performed (pointer path), `false` to fall back to array-based path.
  /// - Parameters:
  ///   - signal: Input signal pointer (flat row-major, inputSize.rows * inputSize.columns elements).
  ///   - filter: Filter kernel pointer (flat row-major, filterSize.rows * filterSize.columns elements).
  ///   - result: Output buffer (caller-allocated, must hold output elements).
  ///   - strides: Convolution strides.
  ///   - padding: Padding mode.
  ///   - filterSize: Kernel shape.
  ///   - inputSize: Input spatial shape.
  /// - Returns: `true` if pointer path was used, `false` to use array-based fallback.
  func conv2dInto(signal: UnsafePointer<Tensor.Scalar>,
                  filter: UnsafePointer<Tensor.Scalar>,
                  result: UnsafeMutablePointer<Tensor.Scalar>,
                  strides: (Int, Int),
                  padding: NumSwift.ConvPadding,
                  filterSize: (rows: Int, columns: Int),
                  inputSize: (rows: Int, columns: Int)) -> Bool
}

extension Device {
  /// Default: pointer path not supported, fall back to array-based conv2d.
  public func conv2dInto(signal: UnsafePointer<Tensor.Scalar>,
                         filter: UnsafePointer<Tensor.Scalar>,
                         result: UnsafeMutablePointer<Tensor.Scalar>,
                         strides: (Int, Int),
                         padding: NumSwift.ConvPadding,
                         filterSize: (rows: Int, columns: Int),
                         inputSize: (rows: Int, columns: Int)) -> Bool {
    false
  }
}

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
  /// The device type identifier (`.cpu` or `.gpu`).
  var type: DeviceType { get }
  /// The Quality of Service class used for concurrent dispatch blocks on this device.
  var qosPriority: DispatchQoS.QoSClass { get set }
  
  /// Computes a transposed 2D convolution directly into `result` using raw pointer storage.
  ///
  /// - Parameters:
  ///   - signal: Pointer to the input feature-map data.
  ///   - filter: Pointer to the transposed-convolution kernel data.
  ///   - result: Pointer to the output buffer where results are written.
  ///   - strides: Row/column stride of the transposed convolution.
  ///   - padding: Padding mode applied to the operation.
  ///   - filterSize: Kernel shape as `(rows, columns)`.
  ///   - inputSize: Input spatial shape as `(rows, columns)`.
  ///   - batchCount: Number of batches to process.
  func transConv2d(signal: TensorStorage.Pointer,
                          filter: TensorStorage.Pointer,
                          result: TensorStorage.Pointer,
                          strides: (Int, Int),
                          padding: NumSwift.ConvPadding,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          batchCount: Int)

  /// Computes a 2D convolution directly into `result` using raw pointer storage.
  ///
  /// - Parameters:
  ///   - signal: Pointer to the input feature-map data.
  ///   - filter: Pointer to the convolution kernel data.
  ///   - result: Pointer to the output buffer where results are written.
  ///   - strides: Row/column stride of the convolution.
  ///   - padding: Padding mode applied to the operation.
  ///   - filterSize: Kernel shape as `(rows, columns)`.
  ///   - inputSize: Input spatial shape as `(rows, columns)`.
  ///   - batchCount: Number of batches to process.
  func conv2d(signal: TensorStorage.Pointer,
              filter: TensorStorage.Pointer,
              result: TensorStorage.Pointer,
              strides: (Int, Int),
              padding: NumSwift.ConvPadding,
              filterSize: (rows: Int, columns: Int),
              inputSize: (rows: Int, columns: Int),
              batchCount: Int)

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
}

final class DeviceManager {
  static let shared = DeviceManager()

  var type: DeviceType = .cpu {
    didSet {
      device = type.device()
    }
  }
  
  private(set) var device: Device = CPU()
}

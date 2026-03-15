//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/13/22.
//

import Foundation
import NumSwift

/// Runs math functions on the CPU
public struct CPU: Device {

  
  /// Priority to run the multithreaded operations on
  public var qosPriority: DispatchQoS.QoSClass = .default
  /// The device type identifier for this implementation, set to `.cpu`.
  public var type: DeviceType = .cpu

  /// Creates a CPU-backed device implementation.
  public init() {}

  /// Performs a transposed 2D convolution (also known as deconvolution) on a batched signal using the CPU.
  /// - Parameter signal: Pointer to the input tensor storage representing the signal.
  /// - Parameter filter: Pointer to the tensor storage representing the convolution filter.
  /// - Parameter result: Pointer to the tensor storage where the output will be written.
  /// - Parameter strides: The stride values as a tuple of (row, column) step sizes.
  /// - Parameter padding: The padding strategy to apply during the transposed convolution.
  /// - Parameter filterSize: The dimensions of the filter as (rows, columns).
  /// - Parameter inputSize: The spatial dimensions of the input as (rows, columns).
  /// - Parameter batchCount: The number of samples in the batch to process.
  public func transConv2d(signal: TensorStorage.Pointer,
                          filter: TensorStorage.Pointer,
                          result: TensorStorage.Pointer,
                          strides: (Int, Int),
                          padding: NumSwift.ConvPadding,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          batchCount: Int) {
    NumSwiftFlat.transConv2dBatch(signal: signal,
                                  filter: filter,
                                  result: result,
                                  strides: strides,
                                  padding: padding,
                                  filterSize: filterSize,
                                  inputSize: inputSize,
                                  batchCount: batchCount)
  }
  
  /// Performs a standard 2D convolution on a batched signal using the CPU.
  /// - Parameter signal: Pointer to the input tensor storage representing the signal.
  /// - Parameter filter: Pointer to the tensor storage representing the convolution filter.
  /// - Parameter result: Pointer to the tensor storage where the output will be written.
  /// - Parameter strides: The stride values as a tuple of (row, column) step sizes.
  /// - Parameter padding: The padding strategy to apply during the convolution.
  /// - Parameter filterSize: The dimensions of the filter as (rows, columns).
  /// - Parameter inputSize: The spatial dimensions of the input as (rows, columns).
  /// - Parameter batchCount: The number of samples in the batch to process.
  public func conv2d(signal: TensorStorage.Pointer,
                     filter: TensorStorage.Pointer,
                     result: TensorStorage.Pointer,
                     strides: (Int, Int),
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     batchCount: Int) {
    NumSwiftFlat.conv2dBatch(signal: signal,
                             filter: filter,
                             result: result,
                             strides: strides,
                             padding: padding,
                             filterSize: filterSize,
                             inputSize: inputSize,
                             batchCount: batchCount)
  }
  
  /// Calculates the convolution of the given inputs
  /// - Parameters:
  ///   - signal: The signal as a Tensor.Value array to perform the convolution on
  ///   - filter: The filter as a Tensor.Value array  to apply to the signal
  ///   - strides: Convolutional strides
  ///   - padding: Convolutional padding
  ///   - filterSize: Size of the filter
  ///   - inputSize: Size of the signal
  ///   - outputSize: Optional declaration of the output size of the convolution
  /// - Returns: The convolution result as a Tensor.Value array
  public func conv2d(signal: Tensor.Value,
                     filter: Tensor.Value,
                     strides: (Int, Int),
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)?) -> Tensor.Value {
    NumSwiftFlat.conv2d(signal: signal,
                        filter: filter,
                        strides: strides,
                        padding: padding,
                        filterSize: filterSize,
                        inputSize: inputSize)
  }
  
  /// Calculates the transposed convolution of the given inputs
  /// - Parameters:
  ///   - signal: The signal as a Tensor.Value to perform the transposed convolution on
  ///   - filter: The filter as a Tensor.Value  to apply to the signal
  ///   - strides: Convolutional strides
  ///   - padding: Convolutional padding
  ///   - filterSize: Size of the filter
  ///   - inputSize: Size of the signal
  ///   - outputSize: Optional declaration of the output size of the transposed convolution
  /// - Returns: The transposed convolution result as a Tensor.Value
  public func transConv2d(signal: Tensor.Value,
                          filter: Tensor.Value,
                          strides: (Int, Int),
                          padding: NumSwift.ConvPadding,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)?) -> Tensor.Value {
    NumSwiftFlat.transConv2d(signal: signal,
                             filter: filter,
                             strides: strides,
                             padding: padding,
                             filterSize: filterSize,
                             inputSize: inputSize)
  }
  
  /// Performs an activation function
  /// - Parameters:
  ///   - input: The input Tensor to perform the activation function on
  ///   - type: The type of activation to apply
  /// - Returns: The activated result as a Tensor
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    type.activate(input: input)
  }
  
  
  /// Performs an activation derivative function
  /// - Parameters:
  ///   - input: The input Tensor to perform the activation derivative function on
  ///   - type: The type of activation derivative to apply
  /// - Returns: The activated derivative result as a Tensor
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    type.derivate(input)
  }
  
  /// Performs matrix multiplication for two tensors.
  ///
  /// - Parameters:
  ///   - a: Left tensor.
  ///   - b: Right tensor.
  /// - Returns: Matrix product tensor.
  public func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
    a.matmul(b)
  }
}

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
  public var type: DeviceType = .cpu

  public init() {}
  
  /// Calculates the transposed convolution of the given inputs
  /// - Parameters:
  ///   - signal: The signal as a 2D array to perform the transposed convolution on
  ///   - filter: The filter as a 2D array  to apply to the signal
  ///   - strides: Convolutional strides
  ///   - padding: Convolutional padding
  ///   - filterSize: Size of the filter
  ///   - inputSize: Size of the signal
  ///   - outputSize: Optional declaration of the output size of the transposed convolution
  /// - Returns: The transposed convolution result as a 2D array
  public func transConv2d(signal: [[Tensor.Scalar]],
                          filter: [[Tensor.Scalar]],
                          strides: (Int, Int) = (1,1),
                          padding: NumSwift.ConvPadding = .valid,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    
    var calculatedOutputSize: (row: Int, columns: Int) {
      var rows = inputSize.rows * strides.0
      var columns = inputSize.columns * strides.1
      
      if padding == .valid {
        rows = (inputSize.rows - 1) * strides.0 + filterSize.rows
        columns = (inputSize.columns - 1) * strides.1 + filterSize.columns
      }

      return (rows, columns)
    }
    
    let result = NumSwiftC.transConv2d(signal: signal,
                                       filter: filter,
                                       strides: strides,
                                       padding: padding,
                                       filterSize: filterSize,
                                       inputSize: inputSize)
    
    return result
  }
  
  /// Calculates the convolution of the given inputs
  /// - Parameters:
  ///   - signal: The signal as a 2D array to perform the convolution on
  ///   - filter: The filter as a 2D array  to apply to the signal
  ///   - strides: Convolutional strides
  ///   - padding: Convolutional padding
  ///   - filterSize: Size of the filter
  ///   - inputSize: Size of the signal
  ///   - outputSize: Optional declaration of the output size of the convolution
  /// - Returns: The convolution result as a 2D array
  public func conv2d(signal: [[Tensor.Scalar]],
                     filter: [[Tensor.Scalar]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    
    var calculatedOutputSize: (row: Int, columns: Int) {
      let paddingValue = padding.extra(inputSize: (inputSize.rows, inputSize.columns), filterSize: filterSize)

      let rows = (((inputSize.rows + (paddingValue.top + paddingValue.bottom)) - (filterSize.rows - 1) - 1) / strides.0) + 1
      let columns = (((inputSize.columns + (paddingValue.left + paddingValue.right)) - (filterSize.columns - 1) - 1) / strides.1) + 1

      return (rows, columns)
    }
    
    let result = NumSwiftC.conv2d(signal: signal,
                                  filter: filter,
                                  strides: strides,
                                  padding: padding,
                                  filterSize: filterSize,
                                  inputSize: inputSize)
    
    return result
  }
  
  /// Performs an activation function
  /// - Parameters:
  ///   - input: The input Tensor to perform the activation function on
  ///   - type: The type of activation to apply
  /// - Returns: The activated result as a Tensor
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    let result = input.value.map { $0.map { $0.map { type.activate(input: $0) }}}
    return Tensor(result)
  }
  
  /// Performs an activation derivative function
  /// - Parameters:
  ///   - input: The input Tensor to perform the activation derivative function on
  ///   - type: The type of activation derivative to apply
  /// - Returns: The activated derivative result as a Tensor
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    let result = input.value.map { $0.map { $0.map { type.derivative(input: $0) }}}
    return Tensor(result)
  }
  
  /// Matrix multiplication operation
  /// - Parameters:
  ///   - a: First matrix
  ///   - b: Second matrix
  ///   - columns: Number of columns (N)
  ///   - rows: Number of rows (K)
  /// - Returns: Resulting matrix multiplication as a Tensor
  @available(*, deprecated, renamed: "matmul", message: "This function has been replaced with `matmul`. It will be removed soon")
  public func matrixMultiply(_ a: Tensor, _ b: Tensor, columns: Int, rows: Int, dimensions: Int = 1) -> Tensor {
    let aFlat: [Tensor.Scalar] = a.value.flatten()
    let bFlat: [Tensor.Scalar] = b.value.flatten()
    
    let multiply = aFlat.multiply(B: bFlat,
                                  columns: Int32(columns),
                                  rows: Int32(rows),
                                  dimensions: Int32(dimensions))
    
    return Tensor(multiply)
  }
  
  public func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
    a.matmul(b)
  }
}

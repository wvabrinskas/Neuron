//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/13/22.
//

import Foundation
import NumSwift

/// Runs math functions on the CPU
public class CPU<N: TensorNumeric>: BaseDevice<N> {
  /// Priority to run the multithreaded operations on

  public override init() {
    super.init()
    qosPriority = .default
    type = .cpu
  }
  
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
  public func transConv2d(signal: [[Tensor<N>.Scalar]],
                                   filter: [[Tensor<N>.Scalar]],
                                   strides: (Int, Int) = (1,1),
                                   padding: NumSwift.ConvPadding = .valid,
                                   filterSize: (rows: Int, columns: Int),
                                   inputSize: (rows: Int, columns: Int),
                                   outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor<N>.Scalar]] where N == Float {
    
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
  public func conv2d(signal: [[Tensor<N>.Scalar]],
                     filter: [[Tensor<N>.Scalar]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor<N>.Scalar]] where N == Float  {
    
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
  ///   - input: The input Tensor<N> to perform the activation function on
  ///   - type: The type of activation to apply
  /// - Returns: The activated result as a Tensor<N>
  public func activate(_ input: Tensor<N>, _ type: Activation) -> Tensor<N> where N == Float {
//    let result = input.value.map { $0.map { $0.map { type.activate(input: $0) }}}
//    return Tensor<N>(result)
    type.activate(input: input)
  }
  
  /// Performs an activation function
  /// - Parameters:
  ///   - input: The input Tensor<N> to perform the activation function on
  ///   - type: The type of activation to apply
  /// - Returns: The activated result as a Tensor<N>
  public func activate(_ input: Tensor<N>, _ type: Activation) -> Tensor<N> where N == Float16 {
//    let result = input.value.map { $0.map { $0.map { type.activate(input: $0) }}}
//    return Tensor<N>(result)
    type.activate(input: input)
  }
  
  
  /// Performs an activation derivative function
  /// - Parameters:
  ///   - input: The input Tensor<N> to perform the activation derivative function on
  ///   - type: The type of activation derivative to apply
  /// - Returns: The activated derivative result as a Tensor<N>
  public func derivate(_ input: Tensor<N>, _ type: Activation) -> Tensor<N> where N == Float  {
    type.derivate(input)
  }
  
  /// Performs an activation derivative function
  /// - Parameters:
  ///   - input: The input Tensor<N> to perform the activation derivative function on
  ///   - type: The type of activation derivative to apply
  /// - Returns: The activated derivative result as a Tensor<N>
  public func derivate(_ input: Tensor<N>, _ type: Activation) -> Tensor<N> where N == Float16  {
    type.derivate(input)
  }
  
  public func matmul(_ a: Tensor<N>, _ b: Tensor<N>) -> Tensor<N> where N == Float {
    a.matmul(b)
  }
  
  public func matmul(_ a: Tensor<N>, _ b: Tensor<N>) -> Tensor<N> where N == Float16 {
    a.matmul(b)
  }
}

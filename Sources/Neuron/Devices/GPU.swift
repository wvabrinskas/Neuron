//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/16/22.
//

import Foundation
import NumSwift

public class GPU: Device {
  public var qosPriority: DispatchQoS.QoSClass = .default
  public var type: DeviceType = .gpu

  private let manager = GPUManager.shared
  
  /// Performs 2D convolution on the GPU device abstraction.
  ///
  /// Currently routed through the flat NumSwift backend while GPU kernels
  /// continue to evolve.
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
  
  /// Performs transposed 2D convolution on the GPU device abstraction.
  ///
  /// Currently routed through the flat NumSwift backend while GPU kernels
  /// continue to evolve.
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
  
  
  /// Applies an activation function using the GPU activation pipeline.
  ///
  /// - Parameters:
  ///   - input: Input tensor.
  ///   - type: Activation to apply.
  /// - Returns: Activated tensor with original shape.
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    let flat = Array(input.storage)
    let activated = self.manager.activate(flat, type)

    let reshaped = activated.reshape(columns: input.size.columns).batched(into: input.size.depth)

    return Tensor(reshaped)
  }
  
  /// Applies an activation derivative using the GPU activation pipeline.
  ///
  /// - Parameters:
  ///   - input: Input tensor.
  ///   - type: Activation derivative to evaluate.
  /// - Returns: Tensor containing derivative values.
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    let flat = Array(input.storage)
    let activated = self.manager.activate(flat, type, derivate: true)

    let reshaped = activated.reshape(columns: input.size.columns).batched(into: input.size.depth)

    return Tensor(reshaped)
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

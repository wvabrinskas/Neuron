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

  private let manager = GPUManager()
  public init() {}

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
    
    let result = NumSwiftC.transConv2d(signal: signal.flatten(),
                                       filter: filter.flatten(),
                                       strides: strides,
                                       padding: padding,
                                       filterSize: filterSize,
                                       inputSize: inputSize)
    
    return result.reshape(columns: outputSize?.columns ?? calculatedOutputSize.columns)
  }
  
  public func conv2d(signal: [[Tensor.Scalar]],
                     filter: [[Tensor.Scalar]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    
    let result = GPUManager().conv2d(signal,
                                     filter: filter,
                                     padding: padding,
                                     filterSize: filterSize,
                                     strides: strides,
                                     inputSize: (inputSize.rows, inputSize.columns, 1))
    
    return result
  }
  
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    let shape = input.shape
      
    let flat = input.value.flatten()
    let activated = GPUManager().activate(flat, type)

    let reshaped = activated.reshape(columns: shape[safe: 0, 0]).batched(into: shape[safe: 2, 0])

    return Tensor(reshaped)
  }
  
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    let shape = input.shape
      
    let flat = input.value.flatten()
    let activated = GPUManager().activate(flat, type, derivate: true)

    let reshaped = activated.reshape(columns: shape[safe: 0, 0]).batched(into: shape[safe: 2, 0])

    return Tensor(reshaped)
  }
  
  public func matrixMultiply(_ a: Tensor, _ b: Tensor, columns: Int, rows: Int) -> Tensor {
    let aFlat: [Tensor.Scalar] = a.value.flatten()
    let bFlat: [Tensor.Scalar] = b.value.flatten()
    
    let multiply = aFlat.multiply(B: bFlat,
                                  columns: Int32(columns),
                                  rows: Int32(rows))
    
    return Tensor(multiply)
  }
}

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
  
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    let shape = input.shape
      
    let flat = input.value.flatten()
    let activated = self.manager.activate(flat, type)

    let reshaped = activated.reshape(columns: shape[safe: 0, 0]).batched(into: shape[safe: 2, 0])

    return Tensor(reshaped)
  }
  
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    let shape = input.shape
      
    let flat = input.value.flatten()
    let activated = self.manager.activate(flat, type, derivate: true)

    let reshaped = activated.reshape(columns: shape[safe: 0, 0]).batched(into: shape[safe: 2, 0])

    return Tensor(reshaped)
  }
  
  @available(*, deprecated, renamed: "matmul", message: "This function has been replaced with `matmul`. It will be removed soon")
  public func matrixMultiply(_ a: Tensor, _ b: Tensor, columns: Int, rows: Int, dimensions: Int) -> Tensor {
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

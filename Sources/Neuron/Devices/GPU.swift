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
  
  public func conv2d(signal: Tensor,
                     filters: [Tensor],
                     biases: [Tensor.Scalar] = [],
                     strides: (Int, Int),
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: TensorSize) -> Tensor {
    
    GPUManager().conv2d(signal,
                        filters: filters,
                        biases: biases,
                        padding: padding,
                        filterSize: filterSize,
                        strides: strides,
                        inputSize: inputSize)
  }
  
  public func conv2d(signal: Tensor,
                     filter: Tensor,
                     strides: (Int, Int),
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: TensorSize) -> Tensor {
    conv2d(signal: signal,
           filters: [filter],
           strides: strides,
           padding: padding,
           filterSize: filterSize,
           inputSize: inputSize)
  }
  
  // MARK: DEPRECATED
  public func conv2d(signal: [[Tensor.Scalar]],
                     filter: [[Tensor.Scalar]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    
    let out = conv2d(signal: Tensor(signal),
                     filters: [Tensor(filter)],
                     strides: strides,
                     padding: padding,
                     filterSize: filterSize,
                     inputSize: TensorSize(rows: inputSize.rows, columns: inputSize.columns, depth: 1))
    
    return out.value.first ?? []
  }
  
  public func activate(_ input: Tensor, _ type: Activation, inputSize: TensorSize) -> Tensor {
    let activated = GPUManager().activate(input,
                                          inputSize: inputSize,
                                          activationType: type)
    return activated
  }
  
  public func derivate(_ input: Tensor, _ type: Activation, inputSize: TensorSize) -> Tensor {
    let activated = GPUManager().activate(input,
                                          inputSize: inputSize,
                                          activationType: type,
                                          derivate: true)
    return activated
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

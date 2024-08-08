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
  
  public init() { }

  public func transConv2d(signal: [[Tensor.Scalar]],
                          filter: [[Tensor.Scalar]],
                          strides: (Int, Int) = (1,1),
                          padding: NumSwift.ConvPadding = .valid,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    
    var calculatedOutputSize: (rows: Int, columns: Int) {
      var rows = inputSize.rows * strides.0
      var columns = inputSize.columns * strides.1
      
      if padding == .valid {
        rows = (inputSize.rows - 1) * strides.0 + filterSize.rows
        columns = (inputSize.columns - 1) * strides.1 + filterSize.columns
      }

      return (rows, columns)
    }
    
    let out = manager.conv2d(input: signal,
                             kernels: filter,
                             strides: strides,
                             padding: padding,
                             filterSize: filterSize,
                             inputSize: inputSize,
                             outputSize: calculatedOutputSize,
                             transConv: true)
    
    return out
  }
  
  public func conv2d(signal: [[Tensor.Scalar]],
                     filter: [[Tensor.Scalar]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    
    var calculatedOutputSize: (rows: Int, columns: Int) {
      let paddingValue = padding.extra(inputSize: (inputSize.rows, inputSize.columns), filterSize: filterSize)

      let rows = (((inputSize.rows + (paddingValue.top + paddingValue.bottom)) - (filterSize.rows - 1) - 1) / strides.0) + 1
      let columns = (((inputSize.columns + (paddingValue.left + paddingValue.right)) - (filterSize.columns - 1) - 1) / strides.1) + 1

      return (rows, columns)
    }
    
    let out = manager.conv2d(input: signal,
                             kernels: filter,
                             strides: strides,
                             padding: padding,
                             filterSize: filterSize,
                             inputSize: inputSize,
                             outputSize: calculatedOutputSize)
    
    return out
  }
  
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    
    let shape = input.shape
    let depth = shape[safe: 2, 0]
    let inputSize = (rows: shape[safe: 1, 0], columns: shape[safe: 0, 0])

    var result: Tensor.Data = []
    
    for d in 0..<depth {
      let activated = manager.activate(to: input.value[d],
                                       inputSize: inputSize,
                                       activationType: type)

      result.append(activated)
    }
    
    return Tensor(result)
  }
  
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    let shape = input.shape
    let depth = shape[safe: 2, 0]
    let inputSize = (rows: shape[safe: 1, 0], columns: shape[safe: 0, 0])

    var result: Tensor.Data = []
    
    for d in 0..<depth {
      let activated = manager.activate(to: input.value[d],
                                       inputSize: inputSize,
                                       activationType: type,
                                       derivate: true)

      result.append(activated)
    }
    
    return Tensor(result)
  }
  
  public func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
    let bShape = b.value.shape
    let aShape = a.value.shape
    let aDepth = aShape[safe: 2] ?? 0

    var result: Tensor.Data = []
    
    for d in 0..<aDepth {
      let cResult = manager.matmul(a.value[d], aShape, b.value[d], bShape)
      result.append(cResult)
    }
    
    return Tensor(result)
  }
}

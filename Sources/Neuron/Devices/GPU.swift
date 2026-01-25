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
    
    func defaultBlock() -> Tensor {
      let shape = input.shape
        
      let flat = input.value.flatten()
      let activated = self.manager.activate(flat, type)

      let reshaped = activated.reshape(columns: shape[safe: 0, 0]).batched(into: shape[safe: 2, 0])

      return Tensor(reshaped)
    }
        
    #if arch(arm64) || arch(x86_64)
    // Use SIMD for small tensors (e.g., early CNN layers)
    if SIMDStrategy.shouldUseSIMD(shape: input.shape) {
      
      let activated: Tensor = switch type {
      case .reLu:
        input.reluSIMD()
      case .tanh:
        input.tanhSIMD()
      case .sigmoid:
        input.sigmoidSIMD()
      case .leakyRelu(let limit):
        input.leakyReluSIMD(limit: limit)
      case .swish:
        input.swishSIMD()
      case .seLu:
        input.seLuSIMD()
      case .geLu:
        input.geLuSIMD()
      default:
        defaultBlock()
      }
      
      return activated
    }
    #endif
    
    return defaultBlock()
  }
  
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    let shape = input.shape
      
    let flat = input.value.flatten()
    let activated = self.manager.activate(flat, type, derivate: true)

    let reshaped = activated.reshape(columns: shape[safe: 0, 0]).batched(into: shape[safe: 2, 0])

    return Tensor(reshaped)
  }
  
  public func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
    a.matmul(b)
  }
}

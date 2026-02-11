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
  
  
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    let flat = Array(input.storage)
    let activated = self.manager.activate(flat, type)

    let reshaped = activated.reshape(columns: input.size.columns).batched(into: input.size.depth)

    return Tensor(reshaped)
  }
  
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    let flat = Array(input.storage)
    let activated = self.manager.activate(flat, type, derivate: true)

    let reshaped = activated.reshape(columns: input.size.columns).batched(into: input.size.depth)

    return Tensor(reshaped)
  }
  
  public func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
    a.matmul(b)
  }
}

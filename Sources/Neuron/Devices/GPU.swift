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
  let metal: NumSwiftMetal = {
    guard let createdMetal = NumSwiftMetal() else {
      fatalError("NumSwiftMetal could not be created")
    }
    return createdMetal
  }()
  
  public init(qosPriority: DispatchQoS.QoSClass = .default) {
    self.qosPriority = qosPriority
  }
  
  public func transConv2d(signal: [[Tensor.Scalar]],
                          filter: [[Tensor.Scalar]],
                          strides: (Int, Int) = (1,1),
                          padding: NumSwift.ConvPadding = .valid,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    metal.transconv2d(signal, filter, stride: strides, padding: padding)
  }
  
  public func conv2d(signal: [[Tensor.Scalar]],
                     filter: [[Tensor.Scalar]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    metal.conv2d(signal, filter, stride: strides, padding: padding)
  }
  
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    var result: [[[Tensor.Scalar]]] = []
    
    for topValue in input.value {
      var row: [[Tensor.Scalar]] = []
      for value in topValue {
        var leakyReluLimit: Tensor.Scalar = 0
        switch type {
        case .leakyRelu(limit: let limit):
          leakyReluLimit = limit
        default:
          break
        }
        
        row.append(metal.activation(value, type: .init(type), limit: leakyReluLimit))
      }
      result.append(row)
    }
  
    return Tensor(result)
  }
  
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    var result: [[[Tensor.Scalar]]] = []
    
    for topValue in input.value {
      var row: [[Tensor.Scalar]] = []
      for value in topValue {
        var leakyReluLimit: Tensor.Scalar = 0
        switch type {
        case .leakyRelu(limit: let limit):
          leakyReluLimit = limit
        default:
          break
        }
        
        row.append(metal.derivative(value, type: .init(type), limit: leakyReluLimit))
      }
      result.append(row)
    }
  
    return Tensor(result)
  }
  
  public func matmul(_ a: Tensor, _ b: Tensor) -> Tensor {
    var result: Tensor.Data = []
    let aShape = TensorSize(array: a.shape)
    let bShape = TensorSize(array: b.shape)
    guard aShape.depth == bShape.depth else {
      fatalError("Matmul failed: incorrect number of channels")
    }
    
    for (i, value) in a.value.enumerated() {
      let aVal = value
      let bVal = b.value[i]
      
      let out = metal.matmul(aVal, bVal)
      
      result.append(out)
    }
    
    return Tensor(result)
  }
}

extension ActivationType {
  
  public init(_ activation: Activation) {
    switch activation {
    case .geLu:
      self = .gelu
    case .leakyRelu:
      self = .leakyRelu
    case .reLu:
      self = .relu
    case .seLu:
      self = .selu
    case .sigmoid:
      self = .sigmoid
    case .swish:
      self = .swish
    case .tanh:
      self = .tanh
    case .softmax:
      self = .softmax
    case .none:
      self = .none
    }
  }
}

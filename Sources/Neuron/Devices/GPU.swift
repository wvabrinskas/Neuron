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
  public var batchSize: Int = 1 {
    didSet {
      commandQueue.processingCount = batchSize
    }
  }

  private let manager = GPUManager.shared
  private let commandQueue = CommandQueue()
  
  let metal: NumSwiftMetal = {
    guard let createdMetal = NumSwiftMetal() else {
      fatalError("NumSwiftMetal could not be created")
    }
    return createdMetal
  }()
    
  public init(qosPriority: DispatchQoS.QoSClass = .default) {
    self.qosPriority = qosPriority
    metal.isAsyncMode = true
    commandQueue.onExecuteQueue = { [metal] in
     // metal.executePendingCommands()
    }
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
    
    let result = commandQueue.enqueue { [metal] result in
      metal.conv2d(signal, filter,
                   stride: strides,
                   padding: padding) { gpuResult in
        result(gpuResult)
      }
    }
  
    return result
  }
  
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    // Flatten entire tensor to 1D array for single GPU call
    let flatInput = input.value.flatMap { $0.flatMap { $0 } }

    // Handle leaky ReLU limit
    var leakyReluLimit: Tensor.Scalar = 0
    if case .leakyRelu(let limit) = type {
      leakyReluLimit = limit
    }

    // Single GPU call for ALL elements at once using NumSwiftMetal
    let flatResult = metal.activation(flatInput, type: .init(type), limit: leakyReluLimit)

    // Reshape [Scalar] back to [[[Scalar]]] using original tensor shape
    let shape = input.shape  // [columns, rows, depth]
    var result: [[[Tensor.Scalar]]] = []
    var index = 0

    for _ in 0..<shape[2] {  // depth
      var depthSlice: [[Tensor.Scalar]] = []
      for _ in 0..<shape[1] {  // rows
        let row = Array(flatResult[index..<(index + shape[0])])  // columns
        depthSlice.append(row)
        index += shape[0]
      }
      result.append(depthSlice)
    }

    return Tensor(result)
  }
  
  public func derivate(_ input: Tensor, _ type: Activation) -> Tensor {
    // Flatten entire tensor to 1D array for single GPU call
    let flatInput = input.value.flatMap { $0.flatMap { $0 } }

    // Handle leaky ReLU limit
    var leakyReluLimit: Tensor.Scalar = 0
    if case .leakyRelu(let limit) = type {
      leakyReluLimit = limit
    }

    // Single GPU call for ALL elements at once using NumSwiftMetal (derivate)
    let flatResult = metal.derivative(flatInput, type: .init(type), limit: leakyReluLimit)

    // Reshape [Scalar] back to [[[Scalar]]] using original tensor shape
    let shape = input.shape  // [columns, rows, depth]
    var result: [[[Tensor.Scalar]]] = []
    var index = 0

    for _ in 0..<shape[2] {  // depth
      var depthSlice: [[Tensor.Scalar]] = []
      for _ in 0..<shape[1] {  // rows
        let row = Array(flatResult[index..<(index + shape[0])])  // columns
        depthSlice.append(row)
        index += shape[0]
      }
      result.append(depthSlice)
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
            
      let out = commandQueue.enqueue { [metal] result in
        metal.matmul(aVal, bVal) { gpuResult in
          result(gpuResult)
        }
      }
            
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

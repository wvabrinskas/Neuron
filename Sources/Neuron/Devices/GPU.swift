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
    metal.isAsyncMode = true
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
  
  @Atomic var iterations: Int = 0
  private let maxIterations = Constants.maxWorkers
  private let condition = NSCondition()

  public func conv2d(signal: [[Tensor.Scalar]],
                     filter: [[Tensor.Scalar]],
                     strides: (Int, Int) = (1,1),
                     padding: NumSwift.ConvPadding = .valid,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)? = nil) -> [[Tensor.Scalar]] {
    // enqueue a X number of operations
    // wait for them to finish
    // move on
    
    // right now this waits for every single one to finish before enqueuing
    condition.lock()
    
    let dispatchGroup = DispatchGroup()

    var result: [[Tensor.Scalar]] = []

    dispatchGroup.enter()

    metal.conv2d(signal, filter,
                 stride: strides,
                 padding: padding) { gpuResult in
      result = gpuResult
      dispatchGroup.leave()
    }
    
    dispatchGroup.wait()

    if result.shape == [0,0] {
      fatalError()
    }
    
    condition.unlock()
    
    iterations = 0

    return result
  }
  
  public func activate(_ input: Tensor, _ type: Activation) -> Tensor {
    // for now perform operation on CPU.
    return CPU().activate(input, type)
    
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
    // for now perform operation on CPU.
    return CPU().derivate(input, type)
    
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
    
    // TODO: after multithreading figure out how to add matmul too
    // this SHOULD work automatically as all operations are sequential but
    // for simplicity sake we're disabling this in the GPU for now until conv is working
    return CPU().matmul(a, b)
    
    var result: Tensor.Data = []
    let aShape = TensorSize(array: a.shape)
    let bShape = TensorSize(array: b.shape)
    guard aShape.depth == bShape.depth else {
      fatalError("Matmul failed: incorrect number of channels")
    }
    
    let bufferSemaphore = DispatchSemaphore(value: 0)

    for (i, value) in a.value.enumerated() {
      let aVal = value
      let bVal = b.value[i]
      
      var out: [[Tensor.Scalar]] = []

      metal.matmul(aVal, bVal) { [bufferSemaphore] gpuResult in
        out = gpuResult
        bufferSemaphore.signal()
      }
      
      bufferSemaphore.wait()
      
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

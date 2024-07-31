//
//  File.swift
//  
//
//  Created by William Vabrinskas on 5/13/22.
//

import Foundation
import NumSwift

public enum DeviceType: String, Codable {
  case cpu, gpu
  
  func device<N: TensorNumeric>() -> BaseDevice<N> {
    switch self {
    case .cpu:
      return CPU<N>()
    case .gpu:
      return GPU<N>()
    }
  }
}

public protocol Device {
  associatedtype N: TensorNumeric
  var type: DeviceType { get }
  var qosPriority: DispatchQoS.QoSClass { get set }
  func conv2d(signal: [[Tensor<N>.Scalar]],
              filter: [[Tensor<N>.Scalar]],
              strides: (Int, Int),
              padding: NumSwift.ConvPadding,
              filterSize: (rows: Int, columns: Int),
              inputSize: (rows: Int, columns: Int),
              outputSize: (rows: Int, columns: Int)?) -> [[Tensor<N>.Scalar]]
  
  func transConv2d(signal: [[Tensor<N>.Scalar]],
                          filter: [[Tensor<N>.Scalar]],
                          strides: (Int, Int),
                          padding: NumSwift.ConvPadding,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)?) -> [[Tensor<N>.Scalar]]

  func activate(_ input: Tensor<N>, _ type: Activation) -> Tensor<N>
  func derivate(_ input: Tensor<N>, _ type: Activation) -> Tensor<N>
  func matmul(_ a: Tensor<N>, _ b: Tensor<N>) -> Tensor<N>
}

open class BaseDevice<N: TensorNumeric>: Device {
  public var type: DeviceType = .cpu
  
  public var qosPriority: DispatchQoS.QoSClass = .userInitiated
  
  public func conv2d(signal: [[Tensor<N>.Scalar]],
                     filter: [[Tensor<N>.Scalar]],
                     strides: (Int, Int),
                     padding: NumSwift.ConvPadding,
                     filterSize: (rows: Int, columns: Int),
                     inputSize: (rows: Int, columns: Int),
                     outputSize: (rows: Int, columns: Int)?) -> [[Tensor<N>.Scalar]] {
    fatalError("Please instantiate a subclass of BaseDevice.")
    return []
  }
  
  public func transConv2d(signal: [[Tensor<N>.Scalar]],
                          filter: [[Tensor<N>.Scalar]],
                          strides: (Int, Int),
                          padding: NumSwift.ConvPadding,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)?) -> [[Tensor<N>.Scalar]] {
    fatalError("Please instantiate a subclass of BaseDevice.")
    return []
  }
  
  public func activate(_ input: Tensor<N>, _ type: Activation) -> Tensor<N> {
    fatalError("Please instantiate a subclass of BaseDevice.")
    return .init()
  }
  
  public func derivate(_ input: Tensor<N>, _ type: Activation) -> Tensor<N> {
    fatalError("Please instantiate a subclass of BaseDevice.")
    return .init()
  }
  
  public func matmul(_ a: Tensor<N>, _ b: Tensor<N>) -> Tensor<N> {
    fatalError("Please instantiate a subclass of BaseDevice.")
    return .init()
  }
  
  
}

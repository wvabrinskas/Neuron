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
  
  func device() -> Device {
    switch self {
    case .cpu:
      return CPU()
    case .gpu:
      return GPU()
    }
  }
}

public protocol Device {
  var type: DeviceType { get }
  var qosPriority: DispatchQoS.QoSClass { get set }
  func conv2d(signal: [[Tensor.Scalar]],
              filter: [[Tensor.Scalar]],
              strides: (Int, Int),
              padding: NumSwift.ConvPadding,
              filterSize: (rows: Int, columns: Int),
              inputSize: (rows: Int, columns: Int),
              outputSize: (rows: Int, columns: Int)?) -> [[Tensor.Scalar]]
  
  func transConv2d(signal: [[Tensor.Scalar]],
                          filter: [[Tensor.Scalar]],
                          strides: (Int, Int),
                          padding: NumSwift.ConvPadding,
                          filterSize: (rows: Int, columns: Int),
                          inputSize: (rows: Int, columns: Int),
                          outputSize: (rows: Int, columns: Int)?) -> [[Tensor.Scalar]]

  func activate(_ input: Tensor, _ type: Activation) -> Tensor
  func derivate(_ input: Tensor, _ type: Activation) -> Tensor
  @available(*, deprecated, renamed: "matmul", message: "This function has been replaced with `matmul`. It will be removed soon")
  func matrixMultiply(_ a: Tensor, _ b: Tensor, columns: Int, rows: Int, dimensions: Int) -> Tensor
  func matmul(_ a: Tensor, _ b: Tensor) -> Tensor
}

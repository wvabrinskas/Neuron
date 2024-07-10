//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/30/22.
//

@testable import Neuron
import Foundation

public enum ColorType: Int, CaseIterable {
  case red, green, blue
  
  public var string: String {
    switch self {
    case .blue:
      return "blue"
    case .red:
      return "red"
    case .green:
      return "green"
    }
  }

  public func correctValues() -> [Tensor.Scalar] {
    var returnArray = [Tensor.Scalar](repeating: 0.0, count: ColorType.allCases.count)
    
    if let index = ColorType.allCases.firstIndex(of: self) {
      returnArray[index] = 1.0
    }
    
    return returnArray
  }
  
  func color() -> [Tensor.Scalar] {
    switch self {
    case .red:
      let r = Tensor.Scalar.random(in: 0.7...1.0)
      let g = Tensor.Scalar.random(in: 0...0.01)
      let b = Tensor.Scalar.random(in: 0...0.01)
      return [r, g, b, 1.0]
    case .green:
      let g = Tensor.Scalar.random(in: 0.7...1.0)
      let r = Tensor.Scalar.random(in: 0...0.01)
      let b = Tensor.Scalar.random(in: 0...0.01)
      return [r, g, b, 1.0]
    case .blue:
      let b = Tensor.Scalar.random(in: 0.7...1.0)
      let r = Tensor.Scalar.random(in: 0...0.01)
      let g = Tensor.Scalar.random(in: 0...0.01)
      return [r, g, b, 1.0]
    }
  }
}

public struct ColorTrainingModel: Equatable {
  var shape: ColorType
  var data: [Tensor.Scalar]
}

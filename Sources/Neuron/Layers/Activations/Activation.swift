//
//  File.swift
//  
//
//  Created by William Vabrinskas on 4/29/22.
//

import Foundation

public enum Activation: Codable, Equatable {
  case reLu
  case sigmoid
  case leakyRelu(limit: Float)
  case swish
  case tanh
  case softmax
  case seLu
  case none
  
  public func index() -> Int {
    switch self {
    case .reLu: return 0
    case .leakyRelu: return 1
    case .sigmoid: return 2
    case .swish: return 3
    case .tanh: return 4
    case .softmax: return 5
    case .seLu: return 6
    case .none: return 7
    }
  }
  
  public func asString() -> String {
    switch self {
    case .reLu: return "reLu"
    case .leakyRelu(let value): return "leakyRelu-\(value)"
    case .sigmoid: return "sigmoid"
    case .swish: return "swish"
    case .tanh: return "tanh"
    case .softmax: return "softmax"
    case .seLu: return "seLu"
    case .none: return "none"
    }
  }
  
  /// Runs the activation function calculation based on the case of self.
  /// - Parameter input: Input Tensor value to run through the activation function
  /// - Returns: The result of the calculation
  public func activate(input: Tensor) -> Tensor {
    let result = input.value.map { $0.map { $0.map { activate(input: $0) }}}
    return Tensor(result)
  }
  
  /// Runs the derivative of the activation function based on the case of self.
  /// - Parameter input: Input Tensor into the calculation
  /// - Returns: The result of the calculation
  public func derivate(_ input: Tensor) -> Tensor {
    let result = input.value.map { $0.map { $0.map { derivative(input: $0) }}}
    return Tensor(result)
  }
  
  /// Runs the activation function calculation based on the case of self.
  /// - Parameter input: Input value to run through the activation function
  /// - Returns: The result of the calculation
  private func activate(input: Float) -> Float {
    var returnValue: Float = 0
    switch self {
    case .seLu:
      let lambda: Float = 1.0507
      let alpha: Float = 1.6733
      if input > 0 {
        return lambda * input
      } else {
        return lambda * alpha * (exp(input) - 1)
      }
    case .reLu:
      returnValue = max(0, input)
    case .sigmoid:
      let out =  1.0 / (1.0 + exp(-input))
      returnValue = out
    case .leakyRelu(let limit):
      if input < 0 {
        returnValue = limit * input
      } else {
        returnValue = input
      }
    case .swish:
      let sigmoid =  1.0 / (1.0 + exp(-input))
      returnValue = input * sigmoid
    case .tanh:
      let x = input
      let num = exp(x) - exp(-x)
      let denom = exp(x) + exp(-x)

      returnValue = num / (denom + 1e-9)
    case .none, .softmax:
      returnValue = input
    }
  
    // filter out completely broken numbers
    guard returnValue.isFinite,
          returnValue.isNormal
    else { return 0 }
    
    return returnValue
  }
  
  /// Runs the derivative of the activation function based on the case of self.
  /// - Parameter input: Input into the calculation
  /// - Returns: The result of the calculation
  private func derivative(input: Float) -> Float {
    switch self {
    case .seLu:
      let lambda: Float = 1.0507
      let alpha: Float = 1.6733
      if input > 0 {
        return lambda
      } else {
        return lambda * alpha * exp(input)
      }
    case .reLu:
      return input >= 0 ? 1 : 0
    case .sigmoid:
      let sig = self.activate(input: input)
      return sig * (1 - sig)
    case .leakyRelu(let limit):
      return input > 0 ? 1 : limit
    case .swish:
      let x = input
      return (exp(-x) * (x + 1) + 1) / pow((1 + exp(-x)), 2)
    case .tanh:
      let tan = self.activate(input: input)
      return 1 - (pow(tan, 2))
    case .none, .softmax:
      return 1
    }
  }
}

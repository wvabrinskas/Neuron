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
  case none
  
  public func index() -> Int {
    switch self {
    case .reLu: return 0
    case .leakyRelu: return 1
    case .sigmoid: return 2
    case .swish: return 3
    case .tanh: return 4
    case .softmax: return 5
    case .none: return 6
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
    case .none: return "none"
    }
  }
  
  /// Runs the activation function calculation based on the case of self.
  /// - Parameter input: Input value to run through the activation function
  /// - Returns: The result of the calculation
  public func activate(input: Float) -> Float {
    var returnValue: Float = 0
    switch self {
    case .reLu:
      returnValue = max(0, input)
    case .sigmoid:
      let out =  1.0 / (1.0 + pow(Float(Darwin.M_E), -input))
      returnValue = out
    case .leakyRelu(let limit):
      if input < 0 {
        returnValue = limit * input
      } else {
        returnValue = input
      }
    case .swish:
      let sigmoid =  1.0 / (1.0 + pow(Float(Darwin.M_E), -input))
      returnValue = input * sigmoid
    case .tanh:
      let e = Float(Darwin.M_E)
      let x = input
      let num = pow(e, x) - pow(e, -x)
      let denom = pow(e, x) + pow(e, -x)

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
  public func derivative(input: Float) -> Float {
    switch self {
    case .reLu:
      return input >= 0 ? 1 : 0
    case .sigmoid:
      let sig = self.activate(input: input)
      return sig * (1 - sig)
    case .leakyRelu(let limit):
      return input > 0 ? 1 : limit
    case .swish:
      let e = Float(Darwin.M_E)
      let x = input
      return (pow(e, -x) * (x + 1) + 1) / pow((1 + pow(e, -x)), 2)
    case .tanh:
      let tan = self.activate(input: input)
      return 1 - (pow(tan, 2))
    case .none, .softmax:
      return 1
    }
  }
}

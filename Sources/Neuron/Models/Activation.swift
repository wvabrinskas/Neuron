//
//  Activation.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/22/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public enum Activation: String, CaseIterable, Codable {
  case reLu
  case sigmoid
  case leakyRelu
  case swish
  case tanh
  case none
  
  /// Runs the activation function calculation based on the case of self.
  /// - Parameter input: Input value to run through the activation function
  /// - Returns: The result of the calculation
  public func activate(input: Float) -> Float {
    switch self {
    case .reLu:
      return max(0, input)
    case .sigmoid:
      let out =  1.0 / (1.0 + pow(Float(Darwin.M_E), -input))
      return out
    case .leakyRelu:
      return max(0.1 * input, input)
    case .swish:
      let sigmoid =  1.0 / (1.0 + pow(Float(Darwin.M_E), -input))
      return input * sigmoid
    case .tanh:
      let e = Float(Darwin.M_E)
      let x = input
      
      let denom = 1 + pow(e, -2 * x)
      return (2 / denom) - 1
    case .none:
      return input
    }
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
    case .leakyRelu:
      return input >= 0 ? 1 : 0.01
    case .swish:
      let e = Float(Darwin.M_E)
      let x = input
      return (pow(e, -x) * (x + 1) + 1) / pow((1 + pow(e, -x)), 2)
    case .tanh:
      let tan = self.activate(input: input)
      return 1 - (pow(tan, 2))
    case .none:
      return 1
    }
  }
  
  /// Gets the activation function name pretty printed
  /// Likely used when generating a UI based on the network. 
  /// - Returns: The name of the activation function based on the case of self.
  public func asString() -> String {
    switch self {
    case .leakyRelu:
      return "Leaky ReLu"
    case .reLu:
      return "ReLu"
    case .sigmoid:
      return "Sigmoid"
    case .swish:
      return "Swish"
    case .tanh:
      return "Tanh"
    case .none:
      return "None"
    }
  }
}

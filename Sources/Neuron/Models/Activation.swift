//
//  Activation.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/22/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public enum Activation: Int, CaseIterable {
  case reLu
  case sigmoid
  case leakyRelu
  case swish
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
      return self.activate(input: input) >= 0 ? 1 : 0
    case .sigmoid:
      let sig = self.activate(input: input)
      return sig * (1 - sig)
    case .leakyRelu:
      return self.activate(input: input) >= 0.01 ? 1 : 0
    case .swish:
      let e = Float(Darwin.M_E)
      let x = input
      return (pow(e, -x) * (x + 1) + 1) / pow((1 + pow(e, -x)), 2)
    case .none:
      return input
    }
  }
  
  /// Gets the activation function name pretty printed
  /// Likely used when generating a UI based on the network. 
  /// - Returns: The name of the activation function based on the case of self.
  public func asString() -> String {
    switch self {
    case .leakyRelu:
      return "Leaky ReLU"
    case .reLu:
      return "ReLU"
    case .sigmoid:
      return "Sigmoid"
    case .swish:
      return "Swish"
    case .none:
      return "None"
    }
  }
}

//
//  Neuron.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/7/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation
import UIKit

public enum Activation {
  case reLu
  case sigmoid
  case leakyRelu
  
  public func activate(input: CGFloat) -> CGFloat {
    switch self {
    case .reLu:
      return max(0, input)
    case .sigmoid:
      return 1.0 / (1.0 + pow(CGFloat(Darwin.M_E), -input))
    case .leakyRelu:
      return max(0.1 * input, input)
    }
  }
}

public class Neuron {
  public var inputs: [NeuroTransmitter] = []
  
  private var learningRate: CGFloat
  private var bias: CGFloat
  private var activationType: Activation
  
  public init(inputs: [NeuroTransmitter] = [], nucleus: Nucleus) {
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = nucleus.activationType
    
    self.inputs = inputs
  }
  
  public func addInput(input: NeuroTransmitter) {
    self.inputs.append(input)
  }
  
  public func get() -> CGFloat {
    return self.activation()
  }
  
  public func activation() -> CGFloat {
    var sum: CGFloat = 0
    for i in 0..<self.inputs.count {
      let input = self.inputs[i]
      let weightInput = input.weight
      sum += weightInput * input.get()
    }
    
    sum += bias
    return activationType.activate(input: sum)
  }
  
  public func updateNucleus(nucleus: Nucleus) {
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = nucleus.activationType
  }
  
  public func clear() {
    for input in inputs {
      input.weight = CGFloat.random(in: 0...1)
    }
  }
  
  public func adjustWeights(correctValue: CGFloat) {
    for input in inputs {
      let activation = self.activation()
      let delta = correctValue - activation
      let correction = self.learningRate * input.get() * delta
      input.weight += correction
      input.neuron?.adjustWeights(correctValue: correctValue)
    }
  }
  
}

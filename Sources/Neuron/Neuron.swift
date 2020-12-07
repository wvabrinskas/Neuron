//
//  Neuron.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/7/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public enum Activation {
  case reLu
  case sigmoid
  case leakyRelu
  
  public func activate(input: Float) -> Float {
    switch self {
    case .reLu:
      return max(0, input)
    case .sigmoid:
      return 1.0 / (1.0 + pow(Float(Darwin.M_E), -input))
    case .leakyRelu:
      return max(0.1 * input, input)
    }
  }
}

public class Neuron {
  public var inputs: [NeuroTransmitter] = []
  
  private var learningRate: Float
  private var bias: Float
  private var activationType: Activation
  
  public init(inputs: [NeuroTransmitter] = [],  nucleus: Nucleus) {
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = nucleus.activationType
    
    self.inputs = inputs
  }
  
  public func replaceInputs(inputs: [NeuroTransmitter]) {
    guard inputs.count == self.inputs.count else {
      print("Error: Can not replace inputs of different size")
      return
    }
    
    var newInputs: [NeuroTransmitter] = []
    
    for i in 0..<self.inputs.count {
      let currentInput = self.inputs[i]
      let newInput = inputs[i]
      
      newInput.weight = currentInput.weight
      newInputs.append(newInput)
    }

    self.inputs = newInputs
  }
  
  public func addInput(input: NeuroTransmitter) {
    self.inputs.append(input)
  }
  
  public func get() -> Float {
    return self.activation()
  }
  
  public func activation() -> Float {
    var sum: Float = 0
    for i in 0..<self.inputs.count {
      let input = self.inputs[i]
      let weightInput = input.weight
      sum += weightInput * input.getNeuronInput()
    }
    
    sum += bias
    print(activationType.activate(input: sum))
    return activationType.activate(input: sum)
  }
  
  public func updateNucleus(nucleus: Nucleus) {
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = nucleus.activationType
  }
  
  public func clear() {
    for input in inputs {
      input.weight = Float.random(in: 0...1)
    }
  }
  
  public func adjustWeights(correctValue: Float) {
    let activation = self.activation()

    DispatchQueue.concurrentPerform(iterations: inputs.count) { (i) in
      let input = inputs[i]
      let delta = correctValue - activation
      let correction = self.learningRate * input.getNeuronInput() * delta
      input.weight += correction
      input.neuron?.adjustWeights(correctValue: correctValue)
    }
  }
  
}

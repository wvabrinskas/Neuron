//
//  Neuron.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/7/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation
import Accelerate

public class Neuron {
  
  /// All inputs connected to this Neuron object
  public var inputs: [NeuroTransmitter] = []
  
  /// Backpropogation delta at this node
  @TestNaN public var delta: Float = 0

  private var learningRate: Float
  private var bias: Float
  private var activationType: Activation
  
  /// Default initializer. Creates a Neuron object
  /// - Parameters:
  ///   - inputs: Array of inputs as NeuroTransmitter that contain the values to be used as inputs
  ///   - nucleus: Nucleus object describing things like learning rate, bias, and activation type
  public init(inputs: [NeuroTransmitter] = [],  nucleus: Nucleus) {
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = nucleus.activationType
    
    self.inputs = inputs
  }
  
  /// Replaces all the inputs connected to this neuron with new ones
  /// - Parameter inputs: Input array as [NeuronTransmitter] to replace inputs iwth
  public func replaceInputs(inputs: [NeuroTransmitter]) {
    if self.inputs.count == 0 {
      self.inputs = inputs
    }
    
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
  
  /// Adds an input as NeuroTransmitter at a specific index.
  /// If the index is greater than the current input count it will just append it to the end of the input array
  /// If there is no index specified it will append it to the end of the input array.
  /// - Parameters:
  ///   - in: NeuroTransmitter to insert
  ///   - index: Optional index to insert the input.
  public func addInput(input in: NeuroTransmitter, at index: Int? = nil) {
    guard let index = index else {
      self.inputs.append(`in`)
      return
    }
    
    if index < self.inputs.count {
      self.inputs[index] = `in`
    } else {
      self.inputs.append(`in`)
    }
  }

  /// Gets the result of the activation function at this node
  /// - Returns: The result of the activation function at this node
  public func activation() -> Float {
    var sum: Float = 0
    
    let inputPointers = self.inputs
    
    for i in 0..<self.inputs.count {
      sum += inputPointers[i].weight * inputPointers[i].inputValue
    }
    
    sum += bias
    
    return self.activationType.activate(input: sum)
  }
  
  
  /// Gets all the inputs as a tuple containing the inputs with their weights
  /// - Returns: A tuple with the inputs as floats and their corresponding weights, in order.
  public func getAllInputs() -> (in: [Float], weight: [Float]) {
    var localInputs: [Float] = []
    var localWeights: [Float] = []

    for i in 0..<self.inputs.count {
      let input = self.inputs[i]
      
      let get = input.inputValue
      let weight = input.weight
      
      localInputs.append(get)
      localWeights.append(weight)
    }
    
    return (localInputs, localWeights)
  }
    
  /// Replaces the Nucleus object describing this Neuron
  /// - Parameter nucleus: Nucleus object to update with
  public func updateNucleus(nucleus: Nucleus) {
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = nucleus.activationType
  }
  
  /// Clears this node of all its weights and
  /// replaces them with a random number between 0 and 1
  public func clear() {
    for input in inputs {
      input.weight = Float.random(in: 0...1)
    }
  }
  
  /// Adjusts the weights of all inputs
  public func adjustWeights() {
    DispatchQueue.concurrentPerform(iterations: inputs.count) { (i) in
      let input = self.inputs[i]
      let activationDer = self.activationType.derivative(input: input.inputValue)

      input.weight += self.learningRate * delta * activationDer  * input.inputValue
    }
  }
  

}


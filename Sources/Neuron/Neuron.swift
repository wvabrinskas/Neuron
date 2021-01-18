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
  public var delta: Float = 0

  private var learningRate: Float
  private var bias: Float
  private var biasWeight: Float = Float.random(in: 0...1)
  private var activationType: Activation
  private var activationDerivative: Float = 0
  private var layer: LobeModel.LayerType
  
  /// Default initializer. Creates a Neuron object
  /// - Parameters:
  ///   - inputs: Array of inputs as NeuroTransmitter that contain the values to be used as inputs
  ///   - nucleus: Nucleus object describing things like learning rate, bias, and activation type
  public init(inputs: [NeuroTransmitter] = [],
              nucleus: Nucleus,
              activation: Activation,
              layer: LobeModel.LayerType) {
    
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = activation
    
    self.inputs = inputs
    self.layer = layer
  }
  
  /// Replaces all the inputs connected to this neuron with new ones
  /// - Parameter inputs: Input array as [Float] to replace inputs with
  /// - Parameter initializer: The initialier to generate the weights

  public func replaceInputs(inputs: [Float], initializer: Inializers = .xavierUniform) {
    if self.inputs.count == 0 {
      
      self.inputs = inputs.map({ (value) -> NeuroTransmitter in
        let t = NeuroTransmitter(input: value)
        let weight = initializer.calculate(m: inputs.count, h: inputs.count)
        t.weight = weight
        return t
      })
    }
    
    guard inputs.count == self.inputs.count else {
      print("Error: Can not replace inputs of different size")
      return
    }
        
    for i in 0..<self.inputs.count {
      self.inputs[i].inputValue = inputs[i]
    }
  }
  
  /// Adds an input as NeuroTransmitter at a specific index.
  /// If the index is greater than the current input count it will just append it to the end of the input array
  /// If there is no index specified it will append it to the end of the input array.
  /// - Parameters:
  ///   - in: NeuroTransmitter to insert
  ///   - index: Optional index to insert the input.
  public func addInput(input in: Float, at index: Int? = nil) {
    guard let index = index, index < self.inputs.count else {
      let neuroTransmitter = NeuroTransmitter(input: `in`)
      self.inputs.append(neuroTransmitter)
      return
    }
    
    self.inputs[index].inputValue = `in`
  }
  
  public func derivative() -> Float {
    return activationDerivative
  }
  /// Gets the result of the activation function at this node. If the layer type is of input
  /// this will return the first input in the array of inputs
  /// - Returns: The result of the activation function at this node
  public func activation() -> Float {
    //if input layer just pass the value along to hidden layer
    if self.layer == .input {
      guard self.inputs.count > 0 else {
        return 0
      }
      
      let input = self.inputs[0].inputValue
      self.activationDerivative = self.activationType.derivative(input: input)
      return input
    }
    
    var sum: Float = 0
        
    //dont add bias or sum of weights to input activation
    for i in 0..<self.inputs.count {
      sum += self.inputs[i].weight * self.inputs[i].inputValue
    }
    
    sum += (bias * biasWeight)
  
    let out = self.activationType.activate(input: sum)
    self.activationDerivative = self.activationType.derivative(input: sum)
    
    return out
  }
  
  /// Replaces the Nucleus object describing this Neuron
  /// - Parameter nucleus: Nucleus object to update with
  public func updateNucleus(nucleus: Nucleus) {
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
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
    //DISPATCH QUEUE BREAKS EVERYTHING NEED BETTER OPTIMIZATION =(
    //WITH OUT IT MAKES IT MUCH SLOWER BUT WITH IT IT FORMS A RACE CONDITION =(
    for i in 0..<inputs.count {
            
      //INVERSE -= to += FOR MSE... ??? idk why
      self.inputs[i].weight -= self.learningRate * self.inputs[i].inputValue * delta * self.derivative()
      biasWeight -= self.learningRate * delta
    }
  }
  

}


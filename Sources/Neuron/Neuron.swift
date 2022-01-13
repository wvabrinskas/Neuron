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
  public var delta: Float? = nil
  public var bias: Float
  public var biasWeight: Float = 0.0
  
  internal var activationType: Activation
  internal var layer: LobeModel.LayerType = .output

  private var learningRate: Float = 0.01
  private var activationDerivative: Float = 0
  private var optimizer: OptimizerFunction?
  
  /// Default initializer. Creates a Neuron object
  /// - Parameters:
  ///   - inputs: Array of inputs as NeuroTransmitter that contain the values to be used as inputs
  ///   - nucleus: Nucleus object describing things like learning rate, bias, and activation type
  public init(inputs: [NeuroTransmitter] = [],
              nucleus: Nucleus,
              activation: Activation,
              optimizer: Optimizer? = nil) {
    
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = activation
    self.inputs = inputs
    self.optimizer = optimizer?.get()
  }
  
  /// Replaces all the inputs connected to this neuron with new ones
  /// - Parameter inputs: Input array as [Float] to replace inputs with
  /// - Parameter initializer: The initialier to generate the weights
  public func addInputs(inputs: [Float], initializer: Initializers = .xavierNormal) {
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
  
    sum += bias * self.biasWeight
  
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
  
  public func gradients() -> [Float] {
    return self.inputs.map { $0.inputValue * (delta ?? 0) * self.derivative() }
  }
  
  /// Adjusts the weights of all inputs
  public func adjustWeights(_ constrain: ClosedRange<Float>? = nil,
                            normalizer: BatchNormalizer? = nil) {
    //DISPATCH QUEUE BREAKS EVERYTHING NEED BETTER OPTIMIZATION =(
    //WITH OUT IT MAKES IT MUCH SLOWER BUT WITH IT IT FORMS A RACE CONDITION =(
    let delta = self.delta ?? 0
    
    var gradients = self.gradients()
    
    if let normalizer = normalizer {
      gradients = normalizer.backward(gradient: gradients)
    }
    
    for i in 0..<gradients.count {
      let gradient = gradients[i]
      
      if let optim = self.optimizer {
        self.inputs[i].weight = optim.run(alpha: self.learningRate,
                                          weight: self.inputs[i].weight,
                                          gradient: gradient)
      } else {
        self.inputs[i].weight -= self.learningRate * gradient
      }
      
      biasWeight -= self.learningRate * delta
      
      if let constrain = constrain {
        let minBound = constrain.lowerBound
        let maxBound = constrain.upperBound
        self.inputs[i].weight = min(maxBound, max(minBound, self.inputs[i].weight))
      }
    }
  }

}


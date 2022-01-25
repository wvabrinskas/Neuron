//
//  Neuron.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/7/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation
import Accelerate
import NumSwift

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
  
  /// Initializes the weights at this neuron using the given initializer
  /// - Parameter count: Number of weights to generate
  /// - Parameter initializer: The initialier to generate the weights
  public func initializeWeights(count: Int, initializer: Initializers = .xavierNormal) {
    if self.inputs.count == 0 {
      
      var newInputs: [NeuroTransmitter] = []
      for _ in 0..<count {
        let t = NeuroTransmitter(input: 0)
        let weight = initializer.calculate(m: count, h: count)
        t.weight = weight
        newInputs.append(t)
      }

      self.inputs = newInputs
    }
  }
  
  /// Replaces all the inputs connected to this neuron with new ones
  /// - Parameter inputs: Input array as [Float] to replace inputs with
  /// - Parameter initializer: The initialier to generate the weights
  public func addInputs(inputs: [Float], initializer: Initializers = .xavierNormal) {
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
  
  /// Applies the activation of a given sum.
  /// - Parameter sum: Dot product of weights * inputs
  /// - Returns: the activated sum given by the activation function at this neuron
  public func applyActivation(sum: Float) -> Float {
    var sum = sum
    sum += bias * self.biasWeight
  
    let out = self.activationType.activate(input: sum)
    self.activationDerivative = self.activationType.derivative(input: sum)
    return out
  }
  
  /// Gets the result of the activation function at this node. If the layer type is of input
  /// this will return the first input in the array of inputs
  /// - Returns: The result of the activation function at this node
  public func activation() -> Float {
    //if input layer just pass the value along to hidden layer
    if self.layer == .input {
      guard let input = self.inputs.first?.inputValue else {
        return 0
      }
      
      self.activationDerivative = self.activationType.derivative(input: input)
      return input
    }
    
    var sum: Float = 0
    //dont add bias or sum of weights to input activation
    for i in 0..<self.inputs.count {
      sum += self.inputs[i].weight * self.inputs[i].inputValue
    }
    
    return self.applyActivation(sum: sum)
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
    let deltaTimeDeriv = self.derivative() * (delta ?? 0)
    return self.inputs.map { $0.inputValue } * deltaTimeDeriv
  }
  
  /// Adjusts the weights of all inputs
  public func adjustWeights(_ constrain: ClosedRange<Float>? = nil) {
    //DISPATCH QUEUE BREAKS EVERYTHING NEED BETTER OPTIMIZATION =(
    //WITH OUT IT MAKES IT MUCH SLOWER BUT WITH IT IT FORMS A RACE CONDITION =(
    let delta = self.delta ?? 0
    
    biasWeight -= self.learningRate * delta
    
    let gradients = self.gradients()

    let group = DispatchGroup()
    
    DispatchQueue.concurrentPerform(iterations: gradients.count) { i in
      group.enter()
      
      let gradient = gradients[i]
      
      if let optim = self.optimizer {
        self.inputs[i].weight = optim.run(alpha: self.learningRate,
                                          weight: self.inputs[i].weight,
                                          gradient: gradient)
      } else {
        self.inputs[i].weight -= self.learningRate * gradient
      }
      
      
      if let constrain = constrain {
        let minBound = constrain.lowerBound
        let maxBound = constrain.upperBound
        self.inputs[i].weight = min(maxBound, max(minBound, self.inputs[i].weight))
      }
      
      group.leave()
    }
    
    group.wait()
  }
}


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
  private(set) var inputValues: [Float] = []
  private(set) var weights: [Float] = []
  
  /// Backpropogation delta at this node
  public var delta: Float? = nil
  public var bias: Float
  public var biasWeight: Float = 0.0
  public var weightGradients: [Float] = []

  internal var activationType: Activation = .none
  internal var layer: LayerType = .output
  internal var activationDerivative: Float = 1
  internal var previousActivation: Float = 0

  private var learningRate: Float = 0.01
  private var optimizer: OptimizerFunction?
  
  /// Default initializer. Creates a Neuron object
  /// - Parameters:
  ///   - inputs: Array of inputs as NeuroTransmitter that contain the values to be used as inputs
  ///   - nucleus: Nucleus object describing things like learning rate, bias, and activation type
  public init(inputs: [NeuroTransmitter] = [],
              nucleus: Nucleus,
              activation: Activation,
              optimizer: OptimizerFunction? = nil) {
    
    self.learningRate = nucleus.learningRate
    self.bias = nucleus.bias
    self.activationType = activation
    self.optimizer = optimizer
    
    for input in inputs {
      self.add(input: input.inputValue, weight: input.weight)
    }
  }
  
  public func addOptimizer(optimizer: OptimizerFunction) {
    self.optimizer = optimizer
  }

  /// Initializes the weights at this neuron
  public func initialize(weights: [Float], inputs: [Float]) {
    guard weights.count == inputs.count else {
      print("Error: Can not replace inputs of different size")
      return
    }
    
    self.weights = weights
    self.inputValues = inputs
    self.weightGradients = [Float].init(repeating: 0, count: self.weights.count)
  }
  
  public func add(input: Float, weight: Float) {
    self.weights.append(weight)
    self.inputValues.append(input)
  }
  
  public func replaceWeights(weights: [Float]) {
    guard weights.count == self.inputValues.count,
          weights.count == self.weights.count else {
      print("Error: Can not replace inputs of different size")
      return
    }
    
    self.weights = weights
  }
  
  /// Replaces all the inputs connected to this neuron with new ones
  /// - Parameter inputs: Input array as [Float] to replace inputs with
  /// - Parameter initializer: The initialier to generate the weights
  public func replaceInputs(inputs: [Float]) {
    guard inputs.count == self.inputValues.count,
          inputs.count == self.weights.count else {
      print("Error: Can not replace inputs of different size")
      return
    }
    
    if layer == .input {
      self.previousActivation = inputs.first ?? 0
    }
    
    self.inputValues = inputs
  }
  
  public func calculateGradients(delta: Float) -> [Float] {
    self.delta = (self.delta ?? 0) + delta
    
    var newGradient = (self.inputValues * delta * activationDerivative)
    newGradient.l2Normalize(limit: 1.0)
    self.weightGradients = self.weightGradients + newGradient
    
    return newGradient
  }
  
  public func zeroGradients() {
    self.delta = nil
    self.weightGradients = [Float].init(repeating: 0, count: self.weights.count)
    self.activationDerivative = 1
    self.previousActivation = 0
  }
  
  /// Applies the activation of a given sum.
  /// - Parameter sum: Dot product of weights * inputs
  /// - Returns: the activated sum given by the activation function at this neuron
  public func applyActivation(sum: Float) -> Float {
    var sum = sum
    sum += bias * self.biasWeight
  
    let out = self.activationType.activate(input: sum)
    self.activationDerivative = self.activationType.derivative(input: sum)
    self.previousActivation = out
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
    //TODO: figure out how to reinitialize this weights with the initilizer
    let clearedWeights = [Float](repeating: Float.random(in: 0...1), count: self.weights.count)
    let clearedInputs = [Float](repeating: 0, count: self.inputValues.count)
    self.initialize(weights: clearedWeights, inputs: clearedInputs)
    self.activationDerivative = 1
    self.previousActivation = 0
  }
  
  /// Adjusts the weights of all inputs
  public func adjustWeights(_ constrain: ClosedRange<Float>? = nil, batchSize: Int) {
    let delta = self.delta ?? 0 / Float(batchSize)
    
    //update bias weight as well using optimizer
    if let optimizer = optimizer {
      biasWeight = optimizer.runBias(weight: biasWeight, gradient: delta)
    } else {
      biasWeight -= self.learningRate * delta
    }
    
    let gradients = self.weightGradients
    
    for i in 0..<gradients.count {
      let gradient = gradients[i] / Float(batchSize) //account for batch size since we append gradients as we back prop
      
      if let optim = self.optimizer {
        self.weights[i] = optim.run(weight: self.weights[i],
                                    gradient: gradient) 
      } else {
        self.weights[i] -= self.learningRate * gradient
      }
      
      if let constrain = constrain {
        let minBound = constrain.lowerBound
        let maxBound = constrain.upperBound
        self.weights[i] = min(maxBound, max(minBound, self.weights[i]))
      }
    }
  }
}


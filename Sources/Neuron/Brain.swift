//
//  Brain.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/6/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

extension Range where Element: Strideable {
  func array() -> [Element] {
    var newArray: [Element] = []
    for i in self {
      newArray.append(i)
    }
    return newArray
  }
}

public class Brain {

  private var neurons: [[Neuron]] = []
  
  /// Creates a Brain object that manages a network of Neuron objects
  /// - Parameters:
  ///   - inputs: Number of inputs to generate
  ///   - outputs: Number of outputs to expect
  ///   - hidden: number of hidden neurons per layer
  ///   - hiddenLayers: number of hidden layers. Default: 1
  ///   - nucleus: Nucleus object that describes the learning behavior of the network
  public init(inputs: Int,
              outputs: Int,
              hidden: Int,
              hiddenLayers: Int = 1,
              nucleus: Nucleus) {
    
    //setup inputs
    var newInputNeurons: [Neuron] = []
    
    for _ in 0..<inputs {
      let inputNeuron = Neuron(nucleus: nucleus)
      newInputNeurons.append(inputNeuron)
    }
    neurons.append(newInputNeurons)
    
    //setup hidden layer
    for _ in 0..<hiddenLayers {
      var newHiddenNeurons: [Neuron] = []
      for _ in 0..<hidden {
        let hiddenNeuron = Neuron(nucleus: nucleus)
        newHiddenNeurons.append(hiddenNeuron)
      }
      neurons.append(newHiddenNeurons)
    }
    
    //setup output layer
    var newOutputNeurons: [Neuron] = []
    for _ in 0..<outputs {
      let outputNeuron = Neuron(nucleus: nucleus)
      newOutputNeurons.append(outputNeuron)
    }
    neurons.append(newOutputNeurons)
    
    //link all the layers generating the matrix
    for i in 0..<neurons.count {
      if i > 0 {
        let neuronGroup = neurons[i]
        let inputNeuronGroup = neurons[i-1]

        let dendrites = inputNeuronGroup.map({ NeuroTransmitter(neuron: $0) })
        neuronGroup.forEach { (neuron) in
          neuron.inputs = dendrites
        }
      }
    }
  
  }
  
  /// Clears the whole network and resets all the weights to a random value
  public func clear() {
    //clears the whole matrix
    self.neurons.forEach { (neuronGroup) in
      neuronGroup.forEach { (neuron) in
        neuron.clear()
      }
    }
  }
  
  /// Feed-forward through the network to get the result
  /// - Parameters:
  ///   - input: the input array of floats
  ///   - ranked: whether the network should sort the output by highest first
  /// - Returns: The result of the feed forward through the network as an array of Floats
  public func feed(input: [Float], ranked: Bool = false) -> [Float] {
  
    self.addInputs(input: input)
    
    for i in 0..<neurons.count {
      if i > 0 {
        let neuronGroup = neurons[i]
        let inputNeuronGroup = neurons[i-1]

        let dendrites = inputNeuronGroup.map({ NeuroTransmitter(neuron: $0) })
        neuronGroup.forEach { (neuron) in
          neuron.replaceInputs(inputs: dendrites)
        }
      }
    }
    
    var outputs: [Float] = []
    
    for output in self.outputLayer() {
      let newOOutput = output.get()
      outputs.append(newOOutput)
    }
    
    let output = ranked ? outputs.sorted(by: { $0 > $1 }) : outputs
    
    return output
  }
  
  /// Get the result of the last layer of the network
  /// - Parameter ranked: whether the network should sort the output by highest first
  /// - Returns: Array of floats resulting from the activation functions of the last layer
  public func get(ranked: Bool = false) -> [Float] {
    
    var outputs: [Float] = []
    
    self.outputLayer().forEach { (neuron) in
      outputs.append(neuron.get())
    }
    
    return ranked ? outputs.sorted(by: { $0 > $1 }) : outputs
  }
  
  
  /// Supervised training function
  /// - Parameters:
  ///   - data: the data to train against as an array of floats
  ///   - correct: the correct value that should be expected from the network
  public func train(data: [Float], correct: Float) {
    self.addInputs(input: data)
    
    DispatchQueue.concurrentPerform(iterations: self.outputLayer().count) { (i) in
      let outNeuron = self.outputLayer()[i]
      outNeuron.adjustWeights(correctValue: correct)
    }
  }
  
  
  /// Train with the data where the output is expected to be the input data
  /// - Parameter data: Input data that contains the expected result
  public func autoTrain(data: [Float]) {
    //not sure we need to add inputs here
    self.addInputs(input: data)
    
    guard data.count == self.outputLayer().count else {
      print("🛑 Error: training data count does not match ouput node count, bailing out")
      return
    }
    
    DispatchQueue.concurrentPerform(iterations: self.outputLayer().count) { (i) in
      let outNeuron = self.outputLayer()[i]
      let value = data[i]
      outNeuron.adjustWeights(correctValue: value)
    }
  }
  
  /// Adds inputs to the input layer where the NeuroTransmitter links a value not a Neuron
  /// - Parameter input: Input array of floats
  private func addInputs(input: [Float]) {
    
    inputLayer().forEach { (inputNeuron) in
      
      if inputNeuron.inputs.count == 0 {
        
        input.forEach { (value) in
          inputNeuron.addInput(input: NeuroTransmitter(input: value))
        }
        
      } else {
        var inputs: [NeuroTransmitter] = []

        input.forEach { (value) in
          inputs.append(NeuroTransmitter(input: value))
        }
        
        inputNeuron.replaceInputs(inputs: inputs)
      }

    }
  }
  
  /// Get first layer of neurons
  /// - Returns: Input layer as array of Neuron objects
  private func inputLayer() -> [Neuron] {
    return self.neurons.first ?? []
  }
  
  /// Get the last layer of neurons
  /// - Returns: Output layer as array of Neuron objects
  private func outputLayer() -> [Neuron] {
    return self.neurons.last ?? []
  }
  
  
}

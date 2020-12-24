//
//  Brain.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/6/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public class Brain {
  
  /// THe nucleus object that describes the network settings
  public var nucleus: Nucleus
  
  /// When set to true this will print the error caclulated by the network as Mean Sqr Error
  public var debug: Bool = false
  
  /// The loss function data from training to be exported
  private var loss: [Float] = []
  
  /// Neuron matrix in the existing brain object
  private var lobes: [Lobe] = []
  
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
    
    self.nucleus = nucleus
    
    //setup inputs
    var newInputNeurons: [Neuron] = []

    for _ in 0..<max(inputs, 1) {
      let inputNeuron = Neuron(nucleus: nucleus)
      newInputNeurons.append(inputNeuron)
    }
    self.lobes.append(Lobe(neurons: newInputNeurons))
        
    //setup hidden layer
    if (hiddenLayers > 0) {
      for _ in 0..<hiddenLayers {
        var newHiddenNeurons: [Neuron] = []
        for _ in 0..<max(hidden, 1) {
          let hiddenNeuron = Neuron(nucleus: nucleus)
          newHiddenNeurons.append(hiddenNeuron)
        }
        self.lobes.append(Lobe(neurons: newHiddenNeurons))
      }
    }
    
    //setup output layer
    var newOutputNeurons: [Neuron] = []
    
    for _ in 0..<max(outputs, 1) {
      let outputNeuron = Neuron(nucleus: nucleus)
      newOutputNeurons.append(outputNeuron)
    }
    self.lobes.append(Lobe(neurons: newOutputNeurons))
    
    //link all the layers generating the matrix
    for i in 0..<lobes.count {
      if i > 0 {
        let neuronGroup = self.lobes[i].neurons
        let inputNeuronGroup = self.lobes[i-1].neurons

        neuronGroup.forEach { (neuron) in
          let dendrites = [NeuroTransmitter](repeating: NeuroTransmitter(), count: inputNeuronGroup.count)
          neuron.inputs = dendrites
        }
      }
    }
  }
  
  /// Train with the data where the output is expected to be the input data
  /// - Parameter data: Input data that contains the expected result
  public func autoTrain(data: [Float]) {
    //not sure we need to add inputs here
    self.train(data: data, correct: data)
  }
  
  /// Supervised training function
  /// - Parameters:
  ///   - data: the data to train against as an array of floats
  ///   - correct: the correct values that should be expected from the network
  public func train(data: [Float], correct: [Float]) {
    guard correct.count == self.outputLayer().count else {
      print("🛑 Error: correct data count does not match ouput node count, bailing out")
      return
    }
    
    //feed the data through the network
    self.feedInternal(input: data)
    
    //get the error and propagate backwards through
    self.backpropagate(correct)
    
    //adjust all the weights after back propagation is complete
    self.adjustWeights()
  }
  
  /// Clears the whole network and resets all the weights to a random value
  public func clear() {
    self.loss.removeAll()
    //clears the whole matrix
    self.lobes.forEach { (lobe) in
      lobe.clear()
    }
  }

  /// Feed-forward through the network to get the result
  /// - Parameters:
  ///   - input: the input array of floats
  ///   - ranked: whether the network should sort the output by highest first
  /// - Returns: The result of the feed forward through the network as an array of Floats
  public func feed(input: [Float], ranked: Bool = false) -> [Float] {
    self.feedInternal(input: input)
    
    let out = self.get(ranked: ranked)
    return out
  }
  
  
  /// Update the settings for each lobe and each neuron
  /// - Parameter nucleus: Object that describes the network behavior
  public func updateNucleus(_ nucleus: Nucleus) {
    self.nucleus = nucleus
    
    self.lobes.forEach { (lobe) in
      lobe.updateNucleus(nucleus)
    }
  }
  
  /// Exports the loss data as a CSV
  /// - Parameter complete: Completion block called when export is finished
  public func exportLoss(_ complete: ((_ url: URL?) -> ())?) {
    let name = "loss-\(Int(Date().timeIntervalSince1970))"
    complete?(ExportManager.getCSV(filename: name, self.loss))
  }
  
  /// Feeds the network internally preparing for output or training
  /// - Parameter input: the input data to feed the network
  private func feedInternal(input: [Float]) {
    self.addInputs(input: input)
    
    for i in 0..<self.lobes.count - 1 {
      let currentLayer = self.lobes[i].neurons
      
      self.lobes[i + 1].neurons.forEach { (neuron) in
        
        let newInputs = currentLayer.map { (neuron) -> NeuroTransmitter in
          return NeuroTransmitter(input: neuron.get())
        }
        
        neuron.replaceInputs(inputs: newInputs)
      }
    }
  }
  
  /// Get the result of the last layer of the network
  /// - Parameter ranked: whether the network should sort the output by highest first
  /// - Returns: Array of floats resulting from the activation functions of the last layer
  private func get(ranked: Bool = false) -> [Float] {
    
    var outputs: [Float] = []
    
    self.outputLayer().forEach { (neuron) in
      outputs.append(neuron.get())
    }
    
    return ranked ? outputs.sorted(by: { $0 > $1 }) : outputs
  }

  /// Adds inputs to the input layer where the NeuroTransmitter links a value
  /// - Parameter input: Input array of floats
  private func addInputs(input: [Float]) {
    guard input.count == self.inputLayer().count else {
      print("🛑 Error: input data count does not match input node count, bailing out")
      return
    }
    
    for i in 0..<inputLayer().count {
      let inputNode = inputLayer()[i]
      let inputValue = input[i]
      
      inputNode.replaceInputs(inputs: [NeuroTransmitter(input: inputValue)])
    }
    
  }
  
  /// Get first layer of neurons
  /// - Returns: Input layer as array of Neuron objects
  private func inputLayer() -> [Neuron] {
    return self.lobes.first?.neurons ?? []
  }
  
  /// Get the last layer of neurons
  /// - Returns: Output layer as array of Neuron objects
  private func outputLayer() -> [Neuron] {
    return self.lobes.last?.neurons ?? []
  }
  
  private func logMeanError(_ calc: Float, correct: Float) {
    let meanErr = calc - correct
    let err = pow(meanErr, 2) / 2
    if debug {
      print(err)
    }
    self.loss.append(err)
  }
  
  private func backpropagate(_ correctValues: [Float]) {
    guard correctValues.count == self.outputLayer().count else {
      print("🛑 Error: correct data count does not match ouput node count, bailing out")
      return
    }
    
    //set output error delta
    for i in 0..<correctValues.count {
      let correct = correctValues[i]
      let outputNeuron = self.outputLayer()[i]
      let get = outputNeuron.get()
      outputNeuron.delta = correct - get
      
      logMeanError(get, correct: correct)
    }
    
    //reverse so we can loop through from the beggining of the array starting at the output node
    let reverse: [Lobe] = self.lobes.reversed()
    
    //- 1 because we dont need to propagate passed the input layer
    for i in 0..<reverse.count - 1 {
      let currentLayer = reverse[i].neurons
      let previousLayer = reverse[i + 1].neurons
      
      for p in 0..<previousLayer.count {
        previousLayer[p].delta = 0
        
        for c in 0..<currentLayer.count {
          let currentNeuronDelta = currentLayer[c].delta * currentLayer[c].inputs[p].weight
          previousLayer[p].delta += currentNeuronDelta
        }
      }

    }
  }
  
  private func adjustWeights() {
    DispatchQueue.concurrentPerform(iterations: self.lobes.count) { (i) in
      let lobe = self.lobes[i]
      lobe.adjustWeights()
    }
  }
}

 

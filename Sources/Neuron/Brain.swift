//
//  Brain.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/6/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public class Brain {
  
  private var currentInput: [Float] = []
  
  private var inputNeurons: [Neuron] = []
  private var outputNeurons: [Neuron] = []
  private var hiddenNeurons: [Neuron] = []
  
  public init(inputs: Int, outputs: Int, hidden: Int, nucleus: Nucleus) {
    
    for _ in 0..<inputs {
      let inputNeuron = Neuron(nucleus: nucleus)
      self.inputNeurons.append(inputNeuron)
    }
    
    for _ in 0..<hidden {
      let hiddenNeuron = Neuron(nucleus: nucleus)
      self.hiddenNeurons.append(hiddenNeuron)
    }
    
    for _ in 0..<outputs {
      let outputNeuron = Neuron(nucleus: nucleus)
      self.outputNeurons.append(outputNeuron)
    }
    
    self.hiddenNeurons.forEach { (neuron) in
      let inputDendrites = self.inputNeurons.map({ return NeuroTransmitter(neuron: $0 )})
      neuron.inputs = inputDendrites
    }
    
    self.outputNeurons.forEach { (neuron) in
      let hiddenDendrites = self.hiddenNeurons.map({ return NeuroTransmitter(neuron: $0 )})
      neuron.inputs = hiddenDendrites
    }
    
  }
  
  public func clear() {
    self.inputNeurons.forEach { (neuron) in
      neuron.clear()
    }
    
    self.hiddenNeurons.forEach { (neuron) in
      neuron.clear()
    }
    
    self.outputNeurons.forEach { (neuron) in
      neuron.clear()
    }
  }
  
  public func getRanked() -> [Float] {
    let output = self.get()
    return output.sorted(by: { $0 > $1 })
  }
  
  public func feed(input: [Float], ranked: Bool = false) -> [Float] {
    self.addInputs(input: input)

    var outputs: [Float] = []

    let inputDendrites = self.inputNeurons.map({ return NeuroTransmitter(neuron: $0 )})
    
    self.hiddenNeurons.forEach { (hNeuron) in
      hNeuron.replaceInputs(inputs: inputDendrites)
    }
        
    let hiddenDendrites = self.hiddenNeurons.map({ return NeuroTransmitter(neuron: $0 )})

    self.outputNeurons.forEach { (oNeuron) in
      oNeuron.replaceInputs(inputs: hiddenDendrites)
      
      let newOOutput = oNeuron.get()
      outputs.append(newOOutput)
    }
    
    let output = ranked ? outputs.sorted(by: { $0 > $1 }) : outputs
    
    return output
  }
  
  public func get() -> [Float] {
    
    var outputs: [Float] = []
    
    self.outputNeurons.forEach { (neuron) in
      outputs.append(neuron.get())
    }
    
    return outputs
  }
  
  public func train(data: [Float], correct: Float) {
    
    self.currentInput = data
    self.addInputs(input: data)
    
    DispatchQueue.concurrentPerform(iterations: self.outputNeurons.count) { (i) in
      let outNeuron = self.outputNeurons[i]
      outNeuron.adjustWeights(correctValue: correct)
    }
  }
  
  public func autoTrain(data: [Float]) {
    
    self.currentInput = data
    self.addInputs(input: data)
    
    guard data.count == self.outputNeurons.count else {
      print("🛑 Error: training data count does not match ouput node count, bailing out")
      return
    }
    
    DispatchQueue.concurrentPerform(iterations: self.outputNeurons.count) { (i) in
      let outNeuron = self.outputNeurons[i]
      let value = data[i]
      outNeuron.adjustWeights(correctValue: value)
    }
  }
  
  private func addInputs(input: [Float]) {
    
    inputNeurons.forEach { (inputNeuron) in
      
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
  
}

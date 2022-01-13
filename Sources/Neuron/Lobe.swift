//
//  Lobe.swift
//  Nameley
//
//  Created by William Vabrinskas on 12/22/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation


/// Class that contains a group of Neurons
public class Lobe {
  
  /// Neurons in the Lobe object
  public var neurons: [Neuron] = []
  public var layer: LobeModel.LayerType = .output
  public var activation: Activation = .none
  public var normalize: Bool = false
  private let normalizer: BatchNormalizer = .init()
  
  /// default initializer
  /// - Parameter neurons: Neruons to control
  public init(neurons: [Neuron],
              activation: Activation = .none,
              normalize: Bool = false) {
    self.neurons = neurons
    self.activation = activation
    self.normalize = normalize
  }
  
  public init(model: LobeModel,
              learningRate: Float) {
    
    let nuc = Nucleus(learningRate: learningRate,
                      bias: model.bias)
    
    var neurons: [Neuron] = []
    for _ in 0..<model.nodes {
      let neuron = Neuron(nucleus: nuc,
                          activation: model.activation)
      neurons.append(neuron)
    }
    self.neurons = neurons
    self.activation = model.activation
    self.normalize = model.normalize
  }
  
  /// Feeds inputs into this layer
  /// - Parameter inputs: inputs into the layer
  /// - Returns: the result of the activation functions of the neurons
  public func feed(inputs: [Float]) -> [Float] {
    var activatedResults: [Float] = []
    
    if self.layer == .input {
      guard inputs.count == self.neurons.count else {
        print("error")
        return []
      }
      
      for i in 0..<inputs.count {
        let input = inputs[i]
        let neuron = neurons[i]
        neuron.addInputs(inputs: [input])
        activatedResults.append(neuron.activation())
      }
      
      return activatedResults
    }
    
    self.neurons.forEach { neuron in
      neuron.addInputs(inputs: inputs)
      activatedResults.append(neuron.activation())
    }
    
    if normalize {
      activatedResults = self.normalizer.normalize(activations: activatedResults)
    }
    
    return activatedResults
  }
  
  /// Sets or updates the current lobes deltas
  /// - Parameters:
  ///   - deltas: the deltas to apply
  ///   - update: boolean to indicate if you add the incoming deltas or just set to the incoming deltas
  public func setLayerDeltas(with deltas: [Float], update: Bool = false) {
    guard deltas.count == neurons.count else {
      return
    }
    
    for i in 0..<neurons.count {
      let delta = deltas[i]
      let neuron = neurons[i]
      
      let updatedDelta = update ? (neuron.delta ?? 0) + delta : delta
      neuron.delta = updatedDelta
    }
  }
  
  /// Calculates deltas for each neuron for the next layer in the network. Updates the current layer deltas with the input previous layer deltas.
  /// - Parameter previousLayerDeltas: Incoming delta's from the previous layer
  /// - Returns: The next set of deltas to be passed to the next layer in the backpropagation.
  public func backpropagate(inputs: [Float], previousLayerCount: Int) -> [Float] {

    guard self.layer != .input else {
      return []
    }
    
    var inputs = inputs
    
    if normalize {
      inputs = normalizer.backward(gradient: inputs)
    }
    
    var deltas: [Float] = []
    //incoming inputs are the new deltas for the current laye
    self.setLayerDeltas(with: inputs, update: true)
    
    for p in 0..<previousLayerCount {
      var delta: Float = 0
      
      for n in 0..<neurons.count {
        let neuron = neurons[n]
        let neuronInput = neuron.inputs[p]
        let neuronDelta = inputs[n]
        
        let currentNeuronDelta = neuronDelta * neuronInput.weight
        
        delta += currentNeuronDelta
      }
      
      deltas.append(delta)
    }

    return deltas
  }
  
  /// Adjusts all the weights in all the neurons in this Lobe
  public func adjustWeights(_ constrain: ClosedRange<Float>? = nil) {
    let normalizer: BatchNormalizer? = self.normalize ? self.normalizer : nil
    
    for neuron in neurons {
      neuron.adjustWeights(constrain, normalizer: normalizer)
    }
  }
  
  /// Clear all the neurons in this Lobe
  public func clear() {
    neurons.forEach { (neuron) in
      neuron.clear()
    }
  }
  
  public func zeroGradients() {
    neurons.forEach { neuron in
      neuron.delta = nil
    }
  }
  
  public func gradients() -> [[Float]] {
    return neurons.map { $0.gradients() }
  }
  
  /// Backpropagation deltas at this specific layer
  /// - Returns: The deltas as floats
  public func deltas() -> [Float] {
    let gradients = neurons.compactMap { $0.delta }
    return gradients
  }
  
  /// Updates the parameters for each Neuron such as learning rate, bias, etc.
  /// - Parameter nucleus: Object that contains the parameters to update
  public func updateNucleus(_ nucleus: Nucleus) {
    neurons.forEach { (neuron) in
      neuron.updateNucleus(nucleus: nucleus)
    }
  }
  
}

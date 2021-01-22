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
  public var layer: LobeModel.LayerType
  public var activation: Activation = .none
    
  /// default initializer
  /// - Parameter neurons: Neruons to control
  public init(neurons: [Neuron],
              layer: LobeModel.LayerType,
              activation: Activation = .none) {
    self.neurons = neurons
    self.layer = layer
    self.activation = activation
  }
  
  public init(model: LobeModel, learningRate: Float) {
    let nuc = Nucleus(learningRate: learningRate,
                      bias: model.bias)

    var neurons: [Neuron] = []
    for _ in 0..<model.nodes {
      let neuron = Neuron(nucleus: nuc,
                          activation: model.activation,
                          layer: model.layer)
      neurons.append(neuron)
    }
    self.neurons = neurons
    self.activation = model.activation
    self.layer = model.layer
  }
  
  /// Adjusts all the weights in all the neurons in this Lobe
  public func adjustWeights() {
    for neuron in neurons {
      neuron.adjustWeights()
    }
  }
  
  /// Clear all the neurons in this Lobe
  public func clear() {
    neurons.forEach { (neuron) in
      neuron.clear()
    }
  }
  
  /// Updates the parameters for each Neuron such as learning rate, bias, etc.
  /// - Parameter nucleus: Object that contains the parameters to update
  public func updateNucleus(_ nucleus: Nucleus) {
    neurons.forEach { (neuron) in
      neuron.updateNucleus(nucleus: nucleus)
    }
  }
}

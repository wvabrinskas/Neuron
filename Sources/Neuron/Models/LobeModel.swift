//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/27/20.
//

import Foundation

/// The model that defines how to construct a Lobe object
public enum LobeModel {
  
  /// The type that the layer should be
  public enum LayerType: CaseIterable {
    case input, hidden, output
  }
  
  case layer(_ nodes: Int, _ activation: Activation = .none, _ layer: LayerType)
  
  //creates a Lobe object with the defining Nucleus
  public func lobe(_ nucleus: Nucleus) -> Lobe {
    switch self {
    case let .layer(nodes, act, layer):
      var nuc = nucleus
      nuc = Nucleus(learningRate: nucleus.learningRate,
                    bias: nucleus.bias)
    
      var neurons: [Neuron] = []
      for _ in 0..<nodes {
        let neuron = Neuron(nucleus: nuc, activation: act, layer: layer)
        neurons.append(neuron)
      }
      return Lobe(neurons: neurons, layer: layer)
    }
  }
}

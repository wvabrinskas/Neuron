//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/27/20.
//

import Foundation

public enum LobeModel {
  
  public enum LayerType: CaseIterable {
    case input, hidden, output
  }
  
  case layer(_ nodes: Int, _ activation: Activation = .none, _ layer: LayerType)
  
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

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/27/20.
//

import Foundation

public enum LobeModel {
  
  case layer(_ nodes: Int, _ activation: Activation? = nil) //a nil activation will use the brain activation
  
  public func lobe(_ nucleus: Nucleus) -> Lobe {
    switch self {
    case let .layer(nodes, act):
      var nuc = nucleus
      if let activation = act {
        nuc = Nucleus(learningRate: nucleus.learningRate,
                      bias: nucleus.bias,
                      activationType: activation)
      }
      var neurons: [Neuron] = []
      for _ in 0..<nodes {
        let neuron = Neuron(nucleus: nuc)
        neurons.append(neuron)
      }
      return Lobe(neurons: neurons)
    }
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/27/20.
//

import Foundation

/// The model that defines how to construct a Lobe object
public struct LobeModel {
  private var nodes: Int
  private var activation: Activation = .none
  private var bias: Float = 0
  private var layer: LayerType
  
  
  /// The type that the layer should be
  public enum LayerType: Int, CaseIterable, Codable {
    case input, hidden, output
  }
  
  //creates a Lobe object with the defining Nucleus
  public init(nodes: Int,
              activation: Activation = .none,
              bias: Float = 0,
              layer: LayerType) {
    self.nodes = nodes
    self.activation = activation
    self.bias = bias
    self.layer = layer
  }
  
  internal func lobe(_ learningRate: Float) -> Lobe {
    let nuc = Nucleus(learningRate: learningRate,
                      bias: bias)
    
    var neurons: [Neuron] = []
    for _ in 0..<nodes {
      let neuron = Neuron(nucleus: nuc, activation: activation, layer: layer)
      neurons.append(neuron)
    }
    return Lobe(neurons: neurons, layer: layer)
  }
}

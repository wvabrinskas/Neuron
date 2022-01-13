//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/27/20.
//

import Foundation

/// The model that defines how to construct a Lobe object
public struct LobeModel {
  internal var nodes: Int
  internal var activation: Activation = .none
  internal var bias: Float = 0
  internal var normalize: Bool = false
  
  /// The type that the layer should be
  public enum LayerType: String, CaseIterable, Codable {
    case input, hidden, output
  }
  
  public init(nodes: Int,
              activation: Activation = .none,
              bias: Float = 0,
              normalize: Bool = false) {
    self.nodes = nodes
    self.activation = activation
    self.bias = bias
    self.normalize = normalize
  }
}

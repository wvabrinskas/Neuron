//
//  LayerNode.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Neuron
import SwiftUI

@available(macOS 14, *)
public class LayerNode: Node {
  
  public var connections: [Node]
  private var layer: EncodingType
  
  init(connections: [Node] = [], layer: EncodingType) {
    self.connections = connections
    self.layer = layer
  }
  
  @ViewBuilder
  public func build() -> any View {
    Text(layer.rawValue)
      .font(.title)
      .padding()
      .background {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
          .fill(Color.gray)
          .stroke(.white, style: .init(lineWidth: 2))
      }
      .padding()
  }
}

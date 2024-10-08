//
//  LayerNode.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Neuron
import SwiftUI

@available(macOS 14, *)
class ActivationNode: BaseNode {
  @ViewBuilder
  override func build() -> any View {
    Text(layer.rawValue)
      .font(.title)
      .foregroundStyle(Color.primary)
      .padding()
      .background {
        RoundedRectangle(cornerRadius: 10, style: .continuous)
          .fill(layer.color)
          .stroke(Color.primary, style: .init(lineWidth: 2))
      }
      .padding()
  }
}

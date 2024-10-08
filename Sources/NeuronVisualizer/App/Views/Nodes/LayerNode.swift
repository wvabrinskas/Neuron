//
//  LayerNode.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Neuron
import SwiftUI

@available(macOS 14, *)
class LayerNode: BaseNode {
  @ViewBuilder
  override func build() -> any View {
    Text(layer.rawValue)
      .font(.title)
      .foregroundStyle(Color.primary)
      .padding()
      .background {
        RoundedRectangle(cornerRadius: 20, style: .continuous)
          .fill(layer.color)
          .stroke(.primary, style: .init(lineWidth: 2))
          .frame(height: 40)
      }
      .padding()
  }
}

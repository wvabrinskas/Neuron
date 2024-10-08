//
//  LayerNode.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Neuron
import SwiftUI

@available(macOS 14, *)
class SizeNode: BaseNode {
  @ViewBuilder
  override func build() -> any View {
    let array = payload.outputSize.asArray
    
    VStack {
      ForEach(array, id: \.self) { int in
        Text("\(int)")
          .fontWeight(.semibold)
      }
    }
    .padding()
    .background {
      RoundedRectangle(cornerRadius: 10, style: .continuous)
        .fill(layer.color)
        .stroke(Color.primary, style: .init(lineWidth: 2))
    }
    .padding()
  }
}


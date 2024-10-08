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
      ForEach(0..<array.count, id: \.self) { int in
        Text("\(array[int])")
          .font(.title2)
          .fontWeight(.semibold)
      }
    }
    .padding()
    .background {
      RoundedRectangle(cornerRadius: 10, style: .continuous)
        .fill(.gray)
        .stroke(Color.primary, style: .init(lineWidth: 2))
    }
    .padding()
  }
}

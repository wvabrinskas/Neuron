//
//  DetailedLayerNode.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Neuron
import SwiftUI

class DetailedActivationLayerNode: BaseNode {
  private let fontSize: CGFloat = 13
  
  @ViewBuilder
  override func build() -> any View {
    Text(layer.rawValue.uppercased())
      .font(.system(size: 16, weight: .bold))
      .foregroundColor(.white)
      .padding(.horizontal, 16)
      .frame(height: 60)
      .background(
        RoundedRectangle(cornerRadius: 30, style: .continuous)
          .stroke(style: StrokeStyle( lineWidth: 4, dash: [5]))
          .fill(layer.color)
      )
  }
}

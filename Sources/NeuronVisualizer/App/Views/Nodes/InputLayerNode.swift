//
//  InputLayerNode.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Neuron
import SwiftUI

@available(macOS 14, *)
class InputLayerNode: BaseNode {
  @ViewBuilder
  override func build() -> any View {
    VStack(alignment: .center, spacing: 0) {
      // Input layer header
      Text("INPUT LAYER")
        .font(.system(size: 16, weight: .bold))
        .foregroundColor(.white)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(Color(red: 0.4, green: 0.7, blue: 0.4))
      
      // Input details
      VStack(alignment: .center, spacing: 6) {
        Text("Shape: \(formatTensorSize(payload.inputSize)) (RGB)")
          .font(.system(size: 11))
          .foregroundColor(.primary)
        
        Text("Parameters: 0")
          .font(.system(size: 11, weight: .medium))
          .foregroundColor(.primary)
      }
      .padding(.horizontal, 16)
      .padding(.vertical, 12)
      .frame(maxWidth: .infinity)
      .background(Color(NSColor.controlBackgroundColor))
    }
    .frame(width: 280)
    .overlay(
      Rectangle()
        .stroke(Color.secondary, lineWidth: 1)
    )
    .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
  }
  
  private func formatTensorSize(_ size: TensorSize) -> String {
    let array = size.asArray
    if array.count <= 1 {
      return "\(array.first ?? 0)"
    }
    return array.map(String.init).joined(separator: "×")
  }
}

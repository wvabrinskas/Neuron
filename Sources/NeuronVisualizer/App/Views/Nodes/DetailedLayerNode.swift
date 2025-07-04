//
//  DetailedLayerNode.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Neuron
import SwiftUI

class DetailedLayerNode: BaseNode {
  private let fontSize: CGFloat = 13
  
  @ViewBuilder
  override func build() -> any View {
    VStack(alignment: .center, spacing: 0) {
      // Layer type header
      Text(getLayerTitle())
        .font(.system(size: 16, weight: .bold))
        .foregroundColor(.white)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 6)
      
      // Layer details
      VStack(alignment: .center, spacing: 6) {
        Text(getLayerDetails())
          .font(.system(size: fontSize))
          .foregroundColor(.primary)
          .multilineTextAlignment(.leading)
        
        // Parameters if available
        if let paramCount = getParameterCount() {
          Text("Parameters: \(paramCount)")
            .font(.system(size: fontSize, weight: .medium))
            .bold()
            .foregroundColor(.primary)
        }
        
      }
      .padding(.horizontal, 16)
      .padding(.vertical, 12)
      .frame(maxWidth: .infinity)
    }
    .frame(width: 280)
    .clipped()
    .background(
      RoundedRectangle(cornerRadius: 20, style: .continuous)
        .fill(layer.color)
    )
  }
  
  private func getLayerTitle() -> String {
    layer.rawValue.uppercased()
  }
  
  private func getLayerDetails() -> String {
    payload.details
  }
  
  private func getParameterCount() -> String? {
    payload.parameters.description
  }
  
  private func formatTensorSize(_ size: TensorSize) -> String {
    let array = size.asArray
    if array.count <= 1 {
      return "\(array.first ?? 0)"
    }
    return array.map(String.init).joined(separator: "×")
  }
  
  private func isOutputLayer() -> Bool {
    // Check if this is likely an output layer (no connections or specific characteristics)
    return connections.isEmpty
  }
}

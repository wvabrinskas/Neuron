//
//  DetailedLayerNode.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import Neuron
import SwiftUI

class DetailedLayerNode: BaseNode {
  @ViewBuilder
  override func build() -> any View {
    VStack(alignment: .center, spacing: 0) {
      // Layer type header
      Text(getLayerTitle())
        .font(.system(size: 16, weight: .bold))
        .foregroundColor(.white)
        .frame(maxWidth: .infinity)
        .padding(.vertical, 12)
        .background(layer.color)
      
      // Layer details
      VStack(alignment: .center, spacing: 6) {
        Text(getLayerDetails())
          .font(.system(size: 11))
          .foregroundColor(.primary)
          .multilineTextAlignment(.leading)
        
        // Parameters if available
        if let paramCount = getParameterCount() {
          Text("Parameters: \(paramCount)")
            .font(.system(size: 11, weight: .medium))
            .foregroundColor(.primary)
        }
        
        // Classes count for final layer
        if layer == .dense && isOutputLayer() {
          Text("Classes: \(payload.outputSize.asArray.last ?? 0)")
            .font(.system(size: 11, weight: .medium))
            .foregroundColor(.primary)
        }
      }
      .padding(.horizontal, 16)
      .padding(.vertical, 12)
      .frame(maxWidth: .infinity)
    }
    .frame(width: 280)
    .overlay(
      RoundedRectangle(cornerRadius: 20, style: .continuous)
        .stroke(Color.secondary, lineWidth: 1)
    )
    .clipped()
  }
  
  private func getLayerTitle() -> String {
    switch layer {
    case .conv2d, .transConv2d:
      return "CONV2D BLOCK"
    case .maxPool, .avgPool:
      return "MAXPOOL2D"
    case .dropout:
      return "DROPOUT"
    case .flatten:
      return "FLATTEN"
    case .dense:
      return "DENSE LAYER"
    case .batchNormalize:
      return "BATCHNORM"
    case .relu, .leakyRelu, .sigmoid, .tanh, .swish, .selu, .softmax:
      return layer.rawValue.uppercased()
    default:
      return layer.rawValue.uppercased()
    }
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

#Preview {
  DetailedLayerNode(payload: .init(layer: .conv2d,
                                   outputSize: .init(array: [32,32,64]),
                                   inputSize: .init(array: [16, 16, 32]),
                                   parameters: 16 * 16 * 32,
                                   details: ""))
  .build()
}

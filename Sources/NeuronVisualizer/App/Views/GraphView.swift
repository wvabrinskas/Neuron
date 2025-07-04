//
//  GraphView.swift
//  Neuron
//
//  Created by William Vabrinskas on 10/7/24.
//

import SwiftUI
import Neuron

public struct GraphView: View {
  let root: Node
    
  public var body: some View {
    ScrollView([.horizontal, .vertical]) {
      VStack(alignment: .center, spacing: 0) {
        let layers = collectLayers(node: root)
        ForEach(0..<layers.count, id: \.self) { i in
          let layer = layers[i]
          
          // Layer block
          AnyView(layer.build())
          
          // Arrow pointing down (except for last layer)
          if i < layers.count - 1 {
            Image(systemName: "arrow.down")
              .font(.title2)
              .foregroundColor(.primary)
              .padding(.vertical, 8)
          }
        }
        
        // Network Summary footer
        VStack(alignment: .center, spacing: 8) {
          Text("Network Summary")
            .font(.system(size: 16, weight: .bold))
            .foregroundColor(.white)
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
            .background(Color(red: 0.3, green: 0.5, blue: 0.8))
          
          VStack(alignment: .leading, spacing: 4) {
            let totalParams = calculateTotalParameters(layers: layers)
            Text("Total Parameters: \(totalParams)")
              .font(.system(size: 12, weight: .medium))
            Text("Layers: \(layers.count - 1)") // Subtract 1 for root node
              .font(.system(size: 12, weight: .medium))
            Text("Input Shape: \(getInputShape(layers: layers))")
              .font(.system(size: 12, weight: .medium))
            Text("Output size: \(getOutputClasses(layers: layers))")
              .font(.system(size: 12, weight: .medium))
          }
          .padding(.horizontal, 16)
          .padding(.vertical, 12)
          .frame(maxWidth: .infinity, alignment: .leading)
          .background(Color(NSColor.controlBackgroundColor))
        }
        .frame(width: 280)
        .overlay(
          Rectangle()
            .stroke(Color.secondary, lineWidth: 1)
        )
        .shadow(color: .black.opacity(0.1), radius: 2, x: 0, y: 1)
        .padding(.top, 16)
      }
      .frame(minWidth: 0, maxWidth: .infinity, minHeight: 0, maxHeight: .infinity)
      .padding()
    }
  }
  
  func collectLayers(node: Node) -> [Node] {
    var layers: [Node] = []
    var currentNode: Node? = node
    
    while let node = currentNode {
      layers.append(node)
      // Move to first connection (assuming sequential network)
      currentNode = node.connections.first
    }
    
    return layers
  }
  
  func calculateTotalParameters(layers: [Node]) -> String {
    // This is a simplified calculation - in practice you'd sum actual layer parameters
    var total = 0
    for layer in layers {
      if let baseNode = layer as? BaseNode {
        total += baseNode.payload.parameters
      }
    }
    return formatNumber(total)
  }
  
  func getInputShape(layers: [Node]) -> String {
    guard layers.count > 1, let firstLayer = layers[1] as? BaseNode else {
      return "Unknown"
    }
    return formatTensorSize(firstLayer.payload.inputSize)
  }
  
  func getOutputClasses(layers: [Node]) -> String {
    guard let lastLayer = layers.last as? BaseNode else {
      return "Unknown"
    }
    return formatTensorSize(lastLayer.payload.outputSize)
  }
  
  func formatTensorSize(_ size: TensorSize) -> String {
    let array = size.asArray
    if array.count <= 1 {
      return "\(array.first ?? 0)"
    }
    return array.map(String.init).joined(separator: "×")
  }
  
  func formatNumber(_ number: Int) -> String {
    let formatter = NumberFormatter()
    formatter.numberStyle = .decimal
    return formatter.string(from: NSNumber(value: number)) ?? "\(number)"
  }
}

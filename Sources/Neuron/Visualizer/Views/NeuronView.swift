//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI

public struct NeuronView: View {
  
  public var viewModel: NeuronViewModel
  
  public init(viewModel: NeuronViewModel) {
    self.viewModel = viewModel
  }
  
  public var body: some View {
    GeometryReader { r in
      ZStack {
        ForEach(0..<viewModel.weights.count, id: \.self) { i in
          let w = viewModel.weights[i]
          path(weight: w, origin: r.frame(in: .local).origin)
            .rotationEffect(Angle(degrees: Double(-90 * i / (viewModel.weights.count - 1))))
        }
      
        Circle()
          .strokeBorder(Color.black, lineWidth: 5)
          .background(Circle().fill(viewModel.color)
            .transition(.opacity)
            .saturation(Double(viewModel.activation))) //expecting 0...1
          .frame(width: viewModel.radius * 2.0,
                 height: viewModel.radius * 2.0)
      }
    }
    .fixedSize()
  }
  
  func path(weight: Float, origin: CGPoint) -> some View {
    Path { path in
      path.move(to: CGPoint(x: origin.x + 5, y: origin.y + 5))
      path.addLine(to: CGPoint(x: self.viewModel.radius, y: self.viewModel.radius))
    }
    .stroke(viewModel.weightsColor, style: StrokeStyle(lineWidth: 10, lineCap: .round, lineJoin: .round))
    .saturation(Double(weight))
  }
}

struct NeuronView_Previews: PreviewProvider {
    static var previews: some View {
      let randomW = Int.random(in: 3...10)
      var weights: [Float] = []
      for _ in 0..<randomW {
        weights.append(Float.random(in: 0...1))
      }
      return NeuronView(viewModel: NeuronViewModel(activation: 0.5, weights: weights))
    }
}

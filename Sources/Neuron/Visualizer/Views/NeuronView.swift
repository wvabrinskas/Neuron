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
    HStack {
      VStack {
        ForEach(0..<viewModel.weights.count) { i in
          let w = viewModel.weights[i]
          Circle()
            .strokeBorder(Color.black, lineWidth: 1)
            .background(Circle().fill(viewModel.color)
              .transition(.opacity)
              .saturation(Double(w)))
            .frame(width: 10,
                   height: 10)
        }
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

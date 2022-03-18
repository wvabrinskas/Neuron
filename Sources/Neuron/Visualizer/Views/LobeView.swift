//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI

public struct LobeView: View {
  
  public var viewModel: LobeViewModel
  
  public init(viewModel: LobeViewModel) {
    self.viewModel = viewModel
  }
  
  public var body: some View {
    VStack(spacing: viewModel.spacing) {
      ForEach(viewModel.neurons) { model in
        NeuronView(viewModel: model)
      }
    }
  }
}

struct LobeView_Previews: PreviewProvider {
  
  static private var neurons: [NeuronViewModel] {
    let num = Int.random(in: 4...10)
    let randomW = Int.random(in: 3...10)

    var models: [NeuronViewModel] = []
    for _ in 0..<num {
      var weights: [Float] = []
      for _ in 0..<randomW {
        weights.append(Float.random(in: 0...1))
      }
      let model = NeuronViewModel(activation: Float.random(in: 0...1),
                                  weights: weights)
      models.append(model)
    }
    return models
  }
  
  static var previews: some View {
    LobeView(viewModel: LobeViewModel(neurons: neurons, spacing: 80))
  }
}

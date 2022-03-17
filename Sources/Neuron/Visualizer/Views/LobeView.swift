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
    let num = Int.random(in: 1...10)
    var models: [NeuronViewModel] = []
    for _ in 0..<num {
      let model = NeuronViewModel(activation: Float.random(in: 0...1))
      models.append(model)
    }
    return models
  }
  
  static var previews: some View {
    LobeView(viewModel: LobeViewModel(neurons: neurons, spacing: 30))
  }
}

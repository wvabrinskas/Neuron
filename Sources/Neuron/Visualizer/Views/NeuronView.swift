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
    Circle()
      .strokeBorder(Color.black, lineWidth: 5)
      .background(Circle().fill(viewModel.color)
        .transition(.opacity)
        .saturation(Double(viewModel.activation))) //expecting 0...1
      .frame(width: viewModel.radius * 2.0,
             height: viewModel.radius * 2.0)
  }
}

struct NeuronView_Previews: PreviewProvider {
    static var previews: some View {
      NeuronView(viewModel: NeuronViewModel(activation: 0.5))
    }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI

public struct VisualizerViewModel {
  var lobes: [LobeViewModel]
}

public struct LobeViewModel {
  var neurons: [NeuronViewModel]
}

public struct NeuronViewModel {
  var activation: Float
}

public struct NeuronView: View {
  
  private var viewModel: NeuronViewModel
  
  public var body: some View {
    Text()
  }
}

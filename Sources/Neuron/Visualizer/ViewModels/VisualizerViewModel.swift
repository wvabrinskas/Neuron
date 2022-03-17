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
  
  public init(lobes: [LobeViewModel]) {
    self.lobes = lobes
  }
}

public struct LobeViewModel {
  var neurons: [NeuronViewModel]
  
  public init(neurons: [NeuronViewModel]) {
    self.neurons = neurons
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI
import NumSwift

public class NetworkVisualizer {
  @Published public var viewModel: VisualizerViewModel?
  private let scale: ClosedRange<Float> = 0...1
  
  public init() {}
  
  public func visualize(brain: Brain) {
    //set view model
    var lobes: [LobeViewModel] = []
    brain.lobes.forEach { l in
      
      let activations = l.neurons.map { $0.previousActivation }.scale(scale)
      
      var models: [NeuronViewModel] = []
      for n in 0..<l.neurons.count {
        let model = NeuronViewModel(activation: activations[n], weights: l.neurons[n].weights.scale(scale))
        models.append(model)
      }

      lobes.append(LobeViewModel(neurons: models))
    }
    
    let brainViewModel = BrainViewModel(lobes: lobes)
    viewModel = VisualizerViewModel(brain: brainViewModel)
  }
}

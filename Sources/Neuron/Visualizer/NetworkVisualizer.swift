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
    var i = 0
    brain.lobes.forEach { l in
      
      let activations = l.neurons.map { $0.previousActivation }.scale(scale)
      
      var models: [NeuronViewModel] = []
      for n in 0..<l.neurons.count {
        let weights = i > 0 ? l.neurons[n].weights.scale(scale) : []
        let model = NeuronViewModel(activation: activations[n], weights: weights)
        models.append(model)
      }

      lobes.append(LobeViewModel(neurons: models))
      i += 1
    }
    
    let brainViewModel = BrainViewModel(lobes: lobes)
    viewModel = VisualizerViewModel(brain: brainViewModel)
  }
}

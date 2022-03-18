//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI
import NumSwift

extension Color {
  static var random: Color {
    return Color(
      red: .random(in: 0...1),
      green: .random(in: 0...1),
      blue: .random(in: 0...1)
    )
  }
}

public class NetworkVisualizer {
  @Published public var viewModel: NetworkViewModel?
  public var randomizeLayerColors: Bool = false
  public var layerColor: Color = .red

  private let scale: ClosedRange<Float> = 0...1
  private var layerColors: [Color] = []
  private var neuronSpacing: CGFloat = 10
  private var layerSpacing: CGFloat = 40
  
  public init(layerColor: Color = .red,
              randomizeLayerColors: Bool = false,
              neuronSpacing: CGFloat = 10,
              layerSpacing: CGFloat = 40) {
    self.layerColor = layerColor
    self.randomizeLayerColors = randomizeLayerColors
    self.neuronSpacing = neuronSpacing
    self.layerSpacing = layerSpacing
  }
  
  public func visualize(brain: Brain) {
    //set view model
    var lobes: [LobeViewModel] = []
    var i = 0
    brain.lobes.forEach { l in
      var color = self.layerColor
      
      if randomizeLayerColors {
        color = layerColors[safe: i] ?? Color.random
        
        if layerColors.count != brain.lobes.count {
          layerColors.append(color)
        }
      }

      let activations = l.neurons.map { $0.previousActivation }.scale(scale)
      
      var models: [NeuronViewModel] = []
      for n in 0..<l.neurons.count {
        let weights = i > 0 ? l.neurons[n].weights.scale(scale) : []
        let model = NeuronViewModel(activation: activations[n],
                                    weights: weights,
                                    color: color,
                                    weightsColor: color)
        models.append(model)
      }

      lobes.append(LobeViewModel(neurons: models, spacing: neuronSpacing))
      i += 1
    }
    
    let brainViewModel = BrainViewModel(lobes: lobes, spacing: layerSpacing)
    viewModel = NetworkViewModel(brain: brainViewModel)
  }
}

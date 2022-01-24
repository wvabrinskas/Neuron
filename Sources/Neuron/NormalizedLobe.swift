//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/13/22.
//

import Foundation

public class NormalizedLobe: Lobe {
  private var normalizer: BatchNormalizer
  public var normalizerLearningParams: (beta: Float, gamma: Float) {
    return (normalizer.beta, normalizer.gamma)
  }
  
  public init(model: LobeModel,
              learningRate: Float,
              batchNormLearningRate: Float,
              beta: Float = 0,
              gamma: Float = 1) {
    
    self.normalizer = BatchNormalizer(gamma: gamma,
                                      beta: beta,
                                      learningRate: batchNormLearningRate)
    
    super.init(model: model, learningRate: learningRate)
    self.isNormalized = true

  }
  
  public init(neurons: [Neuron],
              activation: Activation = .none,
              beta: Float = 0,
              gamma: Float = 1,
              learningRate: Float,
              batchNormLearningRate: Float) {
    
    self.normalizer = BatchNormalizer(gamma: gamma,
                                      beta: beta,
                                      learningRate: batchNormLearningRate)
    
    super.init(neurons: neurons, activation: activation)
    self.isNormalized = true
  }

  public override func feed(inputs: [Float]) -> [Float] {
    let activatedResults = super.feed(inputs: inputs)
    let normalizedResults = self.normalizer.normalize(activations: activatedResults)
    return normalizedResults
  }
  
  public override func backpropagate(inputs: [Float], previousLayerCount: Int) -> [Float] {
    let normalizedBackpropInputs = normalizer.backward(gradient: inputs)
    let backpropResults = super.backpropagate(inputs: normalizedBackpropInputs, previousLayerCount: previousLayerCount)
    return backpropResults
  }
}

//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/13/22.
//

import Foundation

public class NormalizedLobe: Lobe {
  private var normalizer: BatchNormalizer
  public var normalizerLearningParams: (beta: Float,
                                        gamma: Float,
                                        movingMean: Float,
                                        movingVariance: Float) {
    return (normalizer.beta,
            normalizer.gamma,
            normalizer.movingMean,
            normalizer.movingVariance)
  }
  
  public init(model: LobeModel,
              learningRate: Float,
              batchNormLearningRate: Float,
              beta: Float = 0,
              gamma: Float = 1,
              movingMean: Float = 1,
              movingVariance: Float = 1) {
    
    self.normalizer = BatchNormalizer(gamma: gamma,
                                      beta: beta,
                                      learningRate: batchNormLearningRate,
                                      movingMean: movingMean,
                                      movingVariance: movingVariance)
    
    super.init(model: model, learningRate: learningRate)
    self.isNormalized = true

  }
  
  public init(neurons: [Neuron],
              activation: Activation = .none,
              beta: Float = 0,
              gamma: Float = 1,
              learningRate: Float,
              batchNormLearningRate: Float,
              movingMean: Float = 1,
              movingVariance: Float = 1) {
    
    self.normalizer = BatchNormalizer(gamma: gamma,
                                      beta: beta,
                                      learningRate: batchNormLearningRate,
                                      movingMean: movingMean,
                                      movingVariance: movingVariance)
    
    super.init(neurons: neurons, activation: activation)
    self.isNormalized = true
  }

  public override func feed(inputs: [Float], training: Bool) -> [Float] {
    let activatedResults = super.feed(inputs: inputs, training: training)
    let normalizedResults = self.normalizer.normalize(activations: activatedResults, training: training)
    return normalizedResults
  }
  
  public override func backpropagate(inputs: [Float], previousLayerCount: Int) -> [Float] {
    let normalizedBackpropInputs = normalizer.backward(gradient: inputs)
    let backpropResults = super.backpropagate(inputs: normalizedBackpropInputs, previousLayerCount: previousLayerCount)
    return backpropResults
  }
}

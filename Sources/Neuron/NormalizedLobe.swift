//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/13/22.
//

import Foundation

public struct BatchNormalizerParams: Codable {
  public var beta: Float = 0
  public var gamma: Float = 1
  public var movingMean: Float = 1
  public var movingVariance: Float = 1
  public var momentum: Float
  public var learningRate: Float
}

public class NormalizedLobe: Lobe {
  private var normalizer: BatchNormalizer
  public var normalizerLearningParams: BatchNormalizerParams {
    return BatchNormalizerParams(beta: normalizer.beta,
                                 gamma: normalizer.gamma,
                                 movingMean: normalizer.movingMean,
                                 movingVariance: normalizer.movingVariance,
                                 momentum: normalizer.momentum,
                                 learningRate: normalizer.learningRate)
  }
  
  public init(model: LobeModel,
              learningRate: Float,
              momentum: Float,
              batchNormLearningRate: Float,
              beta: Float = 0,
              gamma: Float = 1,
              movingMean: Float = 1,
              movingVariance: Float = 1) {
    
    self.normalizer = BatchNormalizer(gamma: gamma,
                                      beta: beta,
                                      momentum: momentum,
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
              momentum: Float,
              learningRate: Float,
              batchNormLearningRate: Float,
              movingMean: Float = 1,
              movingVariance: Float = 1) {
    
    self.normalizer = BatchNormalizer(gamma: gamma,
                                      beta: beta,
                                      momentum: momentum,
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

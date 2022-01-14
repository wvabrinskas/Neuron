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
              beta: Float = 0,
              gamma: Float = 1) {
    
    self.normalizer = BatchNormalizer(gamma: gamma,
                                      beta: beta,
                                      learningRate: learningRate)
    
    super.init(model: model, learningRate: learningRate)
    self.isNormalized = true

  }
  
  public init(neurons: [Neuron],
              activation: Activation = .none,
              beta: Float = 0,
              gamma: Float = 1,
              learningRate: Float) {
    self.normalizer = BatchNormalizer(gamma: gamma,
                                      beta: beta,
                                      learningRate: learningRate)
    
    super.init(neurons: neurons, activation: activation)
    self.isNormalized = true
  }  

  public override func feed(inputs: [Float]) -> [Float] {
    let normalizedInputs = self.normalizer.normalize(activations: inputs)
    let results = super.feed(inputs: normalizedInputs)
    return results
  }
  
  public override func adjustWeights(_ constrain: ClosedRange<Float>? = nil) {
    for neuron in neurons {
      neuron.adjustWeights(constrain, normalizer: self.normalizer)
    }
  }
  
  public override func backpropagate(inputs: [Float], previousLayerCount: Int) -> [Float] {
    let backpropResults = super.backpropagate(inputs: inputs, previousLayerCount: previousLayerCount)
    let inputs = normalizer.backward(gradient: backpropResults)
    return inputs
  }
}

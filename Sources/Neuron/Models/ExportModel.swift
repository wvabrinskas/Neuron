//
//  File.swift
//  
//
//  Created by William Vabrinskas on 1/20/21.
//

import Foundation


public struct Layer: Codable {
  public var activation: Activation
  public var nodes: Int
  public var weights: [[Float]]
  public var type: LobeModel.LayerType
  public var bias: [Float]
  public var biasWeights: [Float]
  public var normalize: Bool
  public var batchNormalizerParams: BatchNormalizerParams?
  
  public init(activation: Activation,
              nodes: Int,
              weights: [[Float]],
              type: LobeModel.LayerType,
              bias: [Float],
              biasWeights: [Float],
              normalize: Bool,
              batchNormalizerParams: BatchNormalizerParams? = nil) {
    
    self.activation = activation
    self.nodes = nodes
    self.weights = weights
    self.type = type
    self.bias = bias
    self.biasWeights = biasWeights
    self.normalize = normalize
    self.batchNormalizerParams = batchNormalizerParams
  }
}

public struct ExportModel: Codable {
  public var layers: [Layer]
  public var learningRate: Float
  public var optimizer: Optimizer?

  public init(layers: [Layer],
              learningRate: Float,
              optimizer: Optimizer? = nil) {
    self.optimizer = optimizer
    self.layers = layers
    self.learningRate = learningRate
  }
}

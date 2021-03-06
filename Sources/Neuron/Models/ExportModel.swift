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
  
  public init(activation: Activation,
              nodes: Int,
              weights: [[Float]],
              type: LobeModel.LayerType,
              bias: [Float],
              biasWeights: [Float]) {
    
    self.activation = activation
    self.nodes = nodes
    self.weights = weights
    self.type = type
    self.bias = bias
    self.biasWeights = biasWeights
  }
}

public struct ExportModel: Codable {
  public var layers: [Layer]
  public var learningRate: Float
  
  public init(layers: [Layer], learningRate: Float) {
    self.layers = layers
    self.learningRate = learningRate
  }
}

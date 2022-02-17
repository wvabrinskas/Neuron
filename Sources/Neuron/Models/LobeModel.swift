//
//  File.swift
//  
//
//  Created by William Vabrinskas on 12/27/20.
//

import Foundation

/// The type that the layer should be
public enum LayerType: String, CaseIterable, Codable {
  case input, hidden, output
}

public protocol LobeDefinition {
  var nodes: Int { get set }
  var activation: Activation { get set }
  var bias: Float { get set }
}

/// The model that defines how to construct a Lobe object
public struct LobeModel: LobeDefinition {
  public var nodes: Int
  public var activation: Activation = .none
  public var bias: Float = 0

  public init(nodes: Int,
              activation: Activation = .none,
              bias: Float = 0) {
    self.nodes = nodes
    self.activation = activation
    self.bias = bias
  }
}

public struct NormalizedLobeModel: LobeDefinition {
  public var nodes: Int
  public var activation: Activation
  public var bias: Float
  public var momentum: Float
  public var normalizerLearningRate: Float
  
  public init(nodes: Int,
              activation: Activation = .none,
              bias: Float = 0,
              momentum: Float,
              normalizerLearningRate: Float) {
    
    self.nodes = nodes
    self.activation = activation
    self.bias = bias
    self.momentum = momentum
    self.normalizerLearningRate = normalizerLearningRate
  }
}

public struct ConvolutionalLobeModel: LobeDefinition {
  public var nodes: Int
  public var activation: Activation
  public var bias: Float
  public var poolingType: ConvolutionalLobe.PoolType
  
  public init(nodes: Int,
              activation: Activation = .none,
              bias: Float = 0,
              poolingType: ConvolutionalLobe.PoolType) {
    self.bias = bias
    self.activation = activation
    self.nodes = nodes
    self.poolingType = poolingType
  }
}

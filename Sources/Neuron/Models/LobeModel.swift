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
  public var filterSize: (Int, Int)
  public var inputSize: (Int, Int, Int)
  public var filterCount: Int
  
  public init(inputSize: (Int, Int, Int),
              activation: Activation = .none,
              bias: Float = 0,
              filterSize: (Int, Int) = (3, 3),
              filterCount: Int = 1) {
    self.bias = bias
    self.activation = activation
    self.nodes = inputSize.0 * inputSize.1
    self.filterSize = filterSize
    self.inputSize = inputSize
    self.filterCount = filterCount
  }
}

public struct PoolingLobeModel: LobeDefinition {
  public var nodes: Int
  public var activation: Activation
  public var bias: Float
  public var poolingType: PoolingLobe.PoolType
  public var inputSize: (Int, Int)

  public init(inputSize: (Int, Int),
              poolingType: PoolingLobe.PoolType = .max) {
    self.nodes = (inputSize.0 * inputSize.1) / 4
    self.activation = .none
    self.bias = 0
    self.poolingType = poolingType
    self.inputSize = inputSize
  }
}

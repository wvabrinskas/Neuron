//
//  Axon.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/8/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

public struct Nucleus {
  
  public var learningRate: Float = 0.1
  public var bias: Float = 0.1
  
  /// Default initializer for the Nucleus object
  /// - Parameters:
  ///   - learningRate: A float describing the learning rate of the network.
  ///   - bias: A bias to inject into the activation portion of the Neuron.
  public init(learningRate: Float,
              bias: Float) {
    self.learningRate = learningRate
    self.bias = bias
  }
}

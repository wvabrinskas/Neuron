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
  public var activationType: Activation = .reLu
  
  public init(learningRate: Float,
              bias: Float,
              activationType: Activation) {
    self.learningRate = learningRate
    self.bias = bias
    self.activationType = activationType
  }
}

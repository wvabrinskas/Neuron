//
//  Dendrite.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/8/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation
import UIKit


/// Object used to create a link between Neruon objects to act as inputs
public class NeuroTransmitter {
  public weak var neuron: Neuron?
  public var weight: CGFloat
  public var input: CGFloat = 0
  
  //as neuron connection
  public init(neuron: Neuron) {
    self.weight = CGFloat.random(in: 0...1)
    self.neuron = neuron
  }
  
  //as standard input
  public init(input: CGFloat) {
    self.weight = CGFloat.random(in: 0...1)
    self.input = input
  }
  
  public func get() -> CGFloat {
    return self.neuron?.get() ?? self.input
  }
}

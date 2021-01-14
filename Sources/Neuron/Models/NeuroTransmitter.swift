//
//  Dendrite.swift
//  Neuron
//
//  Created by William Vabrinskas on 7/8/20.
//  Copyright © 2020 William Vabrinskas. All rights reserved.
//

import Foundation

/// Object used to create a link between Neruon objects to act as inputs
public class NeuroTransmitter: Equatable {
  private let id: UUID = UUID()
  
  public var weight: Float
  public var inputValue: Float = 0
  
  public static func == (lhs: NeuroTransmitter, rhs: NeuroTransmitter) -> Bool {
    lhs.id == rhs.id
  }
  //as neuron connection
  public init() {
    self.weight = Float.random(in: 0...1)
  }
  
  //as standard input
  public init(input: Float, weight: Float? = nil) {
    self.weight = weight ?? Float.random(in: 0...1)
    self.inputValue = input
  }

}

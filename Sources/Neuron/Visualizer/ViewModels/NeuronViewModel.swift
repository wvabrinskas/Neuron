//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI

public struct NeuronViewModel {
  var activation: Float
  var radius: CGFloat
  var color: Color
  
  public init(activation: Float,
              radius: CGFloat = 10,
              color: Color = .red) {
    self.activation = activation
    self.radius = radius
    self.color = color
  }
}

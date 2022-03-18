//
//  File.swift
//  
//
//  Created by William Vabrinskas on 3/17/22.
//

import Foundation
import SwiftUI

public struct LobeViewModel: Identifiable {
  public var id: UUID = .init()
  var neurons: [NeuronViewModel]
  var spacing: CGFloat
  
  public init(neurons: [NeuronViewModel], spacing: CGFloat = 80) {
    self.neurons = neurons
    self.spacing = spacing
  }
}
